import logging
import pandas as pd
import numpy as np
from multiprocessing import Pool
from collections import defaultdict, Counter, namedtuple
import os
import pickle
import time
import signal
from contextlib import contextmanager

from CIRI_long import env
from CIRI_long.align import *
from CIRI_long.utils import *
from CIRI_long.logger import ProgressBar


LOGGER = logging.getLogger('CIRI-long')
READ = namedtuple('Read', 'read_id circ_id strand cirexon ss clip segments seq sample type')
CIRC = namedtuple('Circ', 'contig start end strand')
globals()['Read'] = READ
globals()['Circ'] = CIRC


class Segment(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.last = None
        self.next = None

    def __str__(self):
        return '{}-{}'.format(self.start, self.end)


class Exon(Segment):
    def __init__(self, start, end):
        self.start = int(start)
        self.end = int(end)


def load_cand_circ(in_file):
    from pathlib import Path
    sample_attr = {}
    with open(in_file, 'r') as f:
        for line in f:
            content = line.rstrip().split()
            if content:
                sample, fname = content
                sample_attr[sample] = fname

    cand_reads = {}
    for sample, fname in sample_attr.items():
        # Full length reads
        cand_circ = Path(fname)
        with open(cand_circ, 'r') as f:
            for line in f:
                content = line.rstrip().lstrip('>').split('\t')
                clip_base = int(content[5].split('|')[1].split('-')[0])
                seq = f.readline().rstrip()
                if clip_base > 20:
                    continue
                cand_reads[content[0]] = READ(*content, seq, sample, 'full')

        # Partial reads
        prefix = cand_circ.name.split('.')[0]
        partial_file = cand_circ.parent / (prefix + '.low_confidence.fa')
        if partial_file.exists():
            with open(partial_file) as f:
                for line in f:
                    content = line.rstrip().lstrip('>').split('\t')
                    clip_base = int(content[5].split('|')[1].split('-')[0])
                    seq = f.readline().rstrip()
                    if clip_base > 20:
                        continue
                    cand_reads[content[0]] = READ(*content, seq, sample, 'partial')

    return cand_reads


def cluster_reads(cand_reads, max_cluster_size=2000):
    """
    Cluster reads by circRNA junction sites with size limits to prevent memory issues
    """
    import re
    from operator import itemgetter

    circ_reads = defaultdict(list)
    circ_start = defaultdict(dict)
    circ_end = defaultdict(dict)

    # Load circRNA junction sites
    for read_id, read in cand_reads.items():
        contig, start, end = re.split('[:-]', read.circ_id)
        start, end = int(start), int(end)
        if end - start > 200 * 1000:
            continue

        # Store reads & BSJ sites
        circ_reads[contig].append((start, end, read.read_id))
        circ_start[contig].setdefault(start, []).append(read.read_id)
        circ_end[contig].setdefault(end, []).append(read.read_id)

    # Build index of circRNA junction sites
    reads_cluster = []
    for contig in circ_reads:
        circ_start_index = {}
        circ_end_index = {}

        # Cluster of start junction site
        tmp = [[], ]
        for x in sorted(circ_start[contig]):
            if not tmp[-1]:
                tmp[-1].append(x)
            elif x > tmp[-1][-1] + 20:
                tmp.append([x, ])
            else:
                tmp[-1].append(x)
        for x in tmp:
            for i in range(min(x) // 500, max(x) // 500 + 1):
                circ_start_index.setdefault(i, []).append(x)

        # Cluster of end junction site
        tmp = [[], ]
        for x in sorted(circ_end[contig]):
            if not tmp[-1]:
                tmp[-1].append(x)
            elif x > tmp[-1][-1] + 20:
                tmp.append([x, ])
            else:
                tmp[-1].append(x)
        for x in tmp:
            for i in range(min(x) // 500, max(x) // 500 + 1):
                circ_end_index.setdefault(i, []).append(x)

        # Cluster reads
        reads_itered = {}
        for (start, end, read_id) in sorted(circ_reads[contig], key=itemgetter(0, 1)):
            if read_id in reads_itered:
                continue

            tmp_reads = []
            p = [i for i in circ_start_index[start // 500] if start in i][0]
            q = [i for i in circ_end_index[end // 500] if end in i][0]

            for i in p:
                tmp_start = circ_start[contig][i]
                for j in q:
                    tmp_end = circ_end[contig][j]
                    tmp = set(tmp_start) & set(tmp_end)
                    if tmp:
                        tmp_reads += tmp

            for i in tmp_reads:
                reads_itered[i] = 1

            # Sort by sequence length
            cluster_reads_list = sorted([cand_reads[i] for i in tmp_reads], 
                                       key=lambda x: len(x.seq), reverse=True)
            
            # Split large clusters to prevent memory issues
            if len(cluster_reads_list) > max_cluster_size:
                LOGGER.info(f'Large cluster detected ({len(cluster_reads_list)} reads), splitting into smaller chunks')
                for chunk_start in range(0, len(cluster_reads_list), max_cluster_size):
                    chunk_end = min(chunk_start + max_cluster_size, len(cluster_reads_list))
                    reads_cluster.append(cluster_reads_list[chunk_start:chunk_end])
            else:
                reads_cluster.append(cluster_reads_list)

    return reads_cluster


def genome_junction_seq(contig, start, end, width=25):
    return env.GENOME.seq(contig, end-width, end) + env.GENOME.seq(contig, start, start + width)


def avg_score(alignment, ref, query):
    x = query[alignment.query_begin:alignment.query_end]
    return distance(ref, x) / len(ref)


def curate_junction(ctg, st, en, junc, max_iterations=1000, stats=None):
    """
    Optimized to reduce search space and prevent excessive iterations
    """
    from libs.striped_smith_waterman.ssw_wrap import Aligner
    from operator import itemgetter
    
    # Reduce search range to be more conservative
    search_window = 25  # keep as 25
    
    # Calculate actual range sizes to avoid overshooting max_iterations
    start_range = min(st) - search_window, max(st) + search_window
    end_range = min(en) - search_window, min(max(en) + search_window, env.CONTIG_LEN[ctg])
    
    start_size = start_range[1] - max(0, start_range[0])
    end_size = end_range[1] - end_range[0]
    total_iterations = start_size * end_size
    
    # If estimated iterations exceed limit, reduce search window further
    if total_iterations > max_iterations:
        scale_factor = (max_iterations / total_iterations) ** 0.5
        search_window = max(5, int(search_window * scale_factor))
    
    junc_scores = []
    iteration_count = 0
    hit_limit = False
    
    for i in range(max(0, min(st) - search_window), max(st) + search_window):
        for j in range(min(en) - search_window, min(max(en) + search_window, env.CONTIG_LEN[ctg])):
            iteration_count += 1
            if iteration_count > max_iterations:
                hit_limit = True
                break
                
            if j <= i:
                continue
            tmp = genome_junction_seq(ctg, i, j, width=10)
            tmp_aligner = Aligner(tmp, match=10, mismatch=4, gap_open=8, gap_extend=2)
            tmp_score = avg_score(tmp_aligner.align(junc), tmp, junc)
            junc_scores.append((i, j, tmp_score))
        if hit_limit:
            break
    
    # Track statistics instead of logging
    if hit_limit and stats is not None:
        stats['curate_junction_limited'] = stats.get('curate_junction_limited', 0) + 1
    
    # Return top 50 results (reduced from 100)
    return sorted(junc_scores, key=itemgetter(2))[:50]


def annotated_hit(contig, scores):
    if env.SS_INDEX is None or contig not in env.SS_INDEX:
        return None

    weighted = []
    for st, en, score in scores:
        w = 0
        if st + 1 in env.SS_INDEX[contig]:
            tmp = set(flatten([p for _, p in env.SS_INDEX[contig][st + 1].items()]))
            if 'start' in tmp:
                w += 1
        elif st in env.SS_INDEX[contig]:
            tmp = set(flatten([p for _, p in env.SS_INDEX[contig][st].items()]))
            if 'end' in tmp:
                w += 1
        else:
            pass

        if en in env.SS_INDEX[contig]:
            tmp = set(flatten([p for _, p in env.SS_INDEX[contig][en].items()]))
            if 'end' in tmp:
                w += 1
        elif en + 1 in env.SS_INDEX[contig]:
            tmp = set(flatten([p for _, p in env.SS_INDEX[contig][en + 1].items()]))
            if 'start' in tmp:
                w += 1
        else:
            pass

        weighted.append([st, en, w])

    return min_sorted_items(weighted, 2, True)


def junc_score(ctg, junc, junc_seqs):
    from libs.striped_smith_waterman.ssw_wrap import Aligner
    aligner = Aligner(env.GENOME.seq(ctg, junc[0], junc[1]) * 2,
                      match=10, mismatch=4, gap_open=8, gap_extend=2)
    score = np.mean([aligner.align(i).score for i in junc_seqs])
    return score


@contextmanager
def time_limit(seconds):
    """Context manager for timing out operations"""
    def signal_handler(signum, frame):
        raise TimeoutError("Operation timeout")
    
    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def correct_chunk(chunk, max_cluster=200, timeout_per_cluster=60):
    """
    Process a chunk of clusters with aggressive timeout protection
    Reduced default timeout from 300s to 60s
    """
    cs_cluster = []
    cnt = defaultdict(int)
    skipped = 0
    timeout_count = 0
    
    # Track optimization events
    stats = {
        'curate_junction_limited': 0,
        'slow_clusters': 0
    }
    
    for idx, cluster in enumerate(chunk):
        if cluster is None:
            continue
        
        cluster_size = len(cluster)
        
        # # Skip extremely large clusters immediately
        # if cluster_size > 200:
            # LOGGER.debug(f'Skipping oversized cluster {idx} with {cluster_size} reads')
            # cnt['skipped_oversized'] = cnt.get('skipped_oversized', 0) + 1
            # continue
        
        try:
            # Use timeout to prevent hanging on problematic clusters
            with time_limit(timeout_per_cluster):
                ret = correct_cluster(cluster, max_cluster=max_cluster, stats=stats)
                if ret is None:
                    continue
                circ_type, circ_attr = ret
                cnt[circ_type] += 1
                cs_cluster.append(circ_attr)
        except TimeoutError:
            LOGGER.warn(f'Timeout: cluster {idx} ({cluster_size} reads, {timeout_per_cluster}s)')
            timeout_count += 1
            continue
        except Exception as e:
            LOGGER.warn(f'Error: cluster {idx} (size {cluster_size}): {str(e)[:100]}')
            skipped += 1
            continue
    
    if skipped > 0:
        cnt['skipped_error'] = skipped
    if timeout_count > 0:
        cnt['skipped_timeout'] = timeout_count
    
    # Add stats to output
    if stats['curate_junction_limited'] > 0:
        cnt['curate_junction_limited'] = stats['curate_junction_limited']
    if stats['slow_clusters'] > 0:
        cnt['slow_clusters'] = stats['slow_clusters']
    
    return cs_cluster, cnt


def correct_cluster(cluster, is_debug=False, max_cluster=200, stats=None):
    """
    Optimized version with early filtering, reduced computational load, and diagnostic timing
    """
    from random import sample
    from collections import Counter
    from spoa import poa
    from libs.striped_smith_waterman.ssw_wrap import Aligner
    import time

    start_time = time.time()
    
    # Early exits
    if cluster is None or len(cluster) <= 1:
        return None
    
    # Filter for full-length reads early
    full_reads = [i for i in cluster if i.type == 'full']
    if not full_reads:
        return None

    original_cluster = cluster[:]
    # Reduce cluster size early to save computation
    if len(full_reads) > max_cluster:
        full_reads = sample(full_reads, max_cluster)
        # Keep some partial reads but limit them
        partial_reads = [i for i in cluster if i.type != 'full']
        # if len(partial_reads) > max_partial_reads:
        partial_reads = sample(partial_reads, int(max_cluster / len(full_reads) * len(partial_reads))) ## ensure consistent proportion
        cluster = full_reads + partial_reads
    
    
    counter = Counter([i.circ_id for i in full_reads]).most_common(n=1)
    ref = sorted([i for i in full_reads if i.circ_id == counter[0][0]],
                 key=lambda x: len(x.seq), reverse=True)[0]
    
    # Use shorter sequence for initial alignment
    ssw = Aligner(ref.seq[:50], match=10, mismatch=4, gap_open=8, gap_extend=2)

    head_pos = []
    valid_queries = []
    
    # Filter queries by alignment quality early
    for query in cluster[1:]:
        alignment = ssw.align(query.seq)
        # Only keep queries with reasonable alignment scores
        if alignment.score > 150:  # Threshold to filter poor alignments
            head_pos.append(alignment.ref_begin)
            valid_queries.append(query)
    
    if not head_pos:
        return None

    # Use filtered queries
    cluster = [ref] + valid_queries
    
    template = transform_seq(ref.seq, max(head_pos))
    ssw = Aligner(template, match=10, mismatch=4, gap_open=8, gap_extend=2)
    junc_seqs = [get_junc_seq(template, -max(head_pos)//2, 25), ]

    for query in valid_queries:
        alignment = ssw.align(query.seq)
        tmp = transform_seq(query.seq, alignment.query_begin)
        junc_seqs.append(get_junc_seq(tmp, -max(head_pos)//2, 25))

    cs_junc, _ = poa(junc_seqs, 2, False, 10, -4, -8, -2, -24, -1)

    ctg = Counter([i.circ_id.split(':')[0] for i in cluster]).most_common()[0][0]
    tmp_st = [int(i.circ_id.split(':')[1].split('-')[0]) for i in cluster]
    tmp_en = [int(i.circ_id.split(':')[1].split('-')[1]) for i in cluster]

    # Curate junction sequence with iteration limit
    scores = curate_junction(ctg, tmp_st, tmp_en, cs_junc, max_iterations=1000, stats=stats)
    
    aval_junc = min_sorted_items(scores, 2)
    if aval_junc:
        anno_junc = annotated_hit(ctg, aval_junc)
        if anno_junc:
            anno_junc = sorted(anno_junc, key=lambda x: junc_score(ctg, x, junc_seqs), reverse=True)
            circ_start, circ_end, circ_score = anno_junc[0]
        else:
            aval_junc = sorted(aval_junc, key=lambda x: junc_score(ctg, x, junc_seqs), reverse=True)
            circ_start, circ_end, circ_score = aval_junc[0]
    else:
        circ_start, circ_end = counter[0][0].split(':')[1].split('-')
        circ_start, circ_end = int(circ_start), int(circ_end)

    # Annotated sites
    ss_site = None
    tmp_signal = None
    for shift_threshold in [5, 10]:
        ss_site, us_free, ds_free, tmp_signal = find_annotated_signal(ctg, circ_start, circ_end, 0, 10, shift_threshold)
        if ss_site is not None:
            ss_id, strand, us_shift, ds_shift = ss_site
            circ_start += us_shift
            circ_end += ds_shift
            circ_type = 'Annotated'
            break

    host_strand = find_host_gene(ctg, circ_start, circ_end)
    circ_type = None

    # Canonical sites
    if ss_site is None:
        for shift_threshold in [5, 10]:
            ss_site = find_denovo_signal(ctg, circ_start, circ_end, host_strand, tmp_signal,
                                         us_free, ds_free, 0, 10, shift_threshold, True)
            if ss_site is not None:
                ss_id, strand, us_shift, ds_shift = ss_site
                circ_start += us_shift
                circ_end += ds_shift
                circ_type = 'Annotated'
                break

    # Intronic circRNAs
    if ss_site is None:
        retained_introns = find_retained_introns(ctg, circ_start + 1, circ_end)
        overlap_exons = find_overlap_exons(ctg, circ_start + 1, circ_end)

        is_lariat = 0
        if retained_introns is not None and overlap_exons is None:
            is_lariat = 1
            retained_introns = set(sum([i for _, i in retained_introns.items()], []))
            retained_strand = set([i[2] for i in retained_introns])
            tmp_circ = []
            for intron_start, intron_end, intron_strand in retained_introns:
                if abs(intron_start - circ_start) > 50 or abs(intron_end - circ_end) > 50:
                    continue
                if intron_strand == '+':
                    tmp_site = [i for i in scores if i[0] == intron_start]
                else:
                    tmp_site = [i for i in scores if i[1] == intron_end]
                if tmp_site:
                    tmp_circ.append([*tmp_site[0], intron_strand])

            ss_id = 'lariat'
            if tmp_circ:
                circ_start, circ_end, circ_score, strand = sorted(tmp_circ, key=lambda x: x[2])[0]
                circ_type = 'High confidence lariat'
            else:
                is_lariat = 0
                tmp_circ = []
                for tmp_strand in retained_strand:
                    tmp_start, tmp_end, tmp_score = recursive_splice_site(scores, ctg, tmp_strand)
                    if tmp_score is not None:
                        tmp_circ.append([tmp_start, tmp_end, tmp_score, tmp_strand])
                if tmp_circ:
                    circ_start, circ_end, circ_score, strand = sorted(tmp_circ, key=lambda x: x[2])[0]
                else:
                    strand = 'None'

        # Find denovo splice signal
        if is_lariat == 0:
            ss_site = find_denovo_signal(ctg, circ_start, circ_end, host_strand, tmp_signal,
                                         us_free, ds_free, 5, 10, 3, False)
            if ss_site is not None:
                ss_id, strand, us_shift, ds_shift = ss_site
                circ_start += us_shift
                circ_end += ds_shift
                circ_type = 'Denovo signal'
            else:
                ss_id = 'None'
                strand = 'None'
                circ_type = 'Unknown signal'

    circ_id = '{}:{}-{}'.format(ctg, circ_start + 1, circ_end)

    # refined sequence
    cluster_seq = []
    circ_junc_seq = genome_junction_seq(ctg, circ_start, circ_end)
    ssw = Aligner(circ_junc_seq, match=10, mismatch=4, gap_open=8, gap_extend=2, report_cigar=True)

    tmp_cluster = [i for i in cluster if i.type == 'full']
    if len(tmp_cluster) > max_cluster:
        tmp_cluster = sample(tmp_cluster, max_cluster)
    tmp_cluster = sorted(tmp_cluster, key=lambda x: len(x.seq), reverse=True)

    for query in tmp_cluster:
        alignment = ssw.align(query.seq * 2)
        tmp_pos = find_alignment_pos(alignment, len(circ_junc_seq)//2)
        if tmp_pos is None:
            cluster_seq.append((query.read_id, query.seq))
        else:
            tmp_seq = transform_seq(query.seq, tmp_pos % len(query.seq))
            cluster_seq.append((query.read_id, tmp_seq))

    cluster_res = batch_cluster_sequence(circ_id, cluster_seq, max_attempts=10)
    cluster_res = sorted(cluster_res, key=lambda x: len(x[1]), reverse=True)

    circ = CIRC(ctg, circ_start + 1, circ_end, strand)
    circ_id = '{}:{}-{}'.format(circ.contig, circ.start, circ.end)

    if len(cluster_res) > 2 and len(cluster_res[0][1]) >= 0.5 * max(len(tmp_cluster), 10):
        tmp_res = correct_cluster([i for i in cluster if i.read_id in cluster_res[0][1]], True, max_cluster, stats)
        if tmp_res is not None:
            circ = tmp_res
            circ_id = '{}:{}-{}'.format(circ.contig, circ.start, circ.end)

    # Filter out strange cirexons
    curated_exons = curate_cirexons(circ, original_cluster)
    if curated_exons is None:
        return None
    isoforms, isoform_reads, circ_len = curate_isoform(circ, curated_exons, cluster_res)
    if isoforms is None:
        return None
    is_concordance = check_isoforms(circ, isoforms)
    if not is_concordance:
        return None

    total_time = time.time() - start_time
    if total_time > 30:
        if stats is not None:
            stats['slow_clusters'] = stats.get('slow_clusters', 0) + 1

    if is_debug:
        return circ

    return circ_type, ([i.read_id for i in original_cluster], isoform_reads, cluster_seq, circ_id, circ.strand,
                       ss_id, us_free, ds_free, circ_len, isoforms)


def batch_cluster_sequence(circ_id, x, max_attempts=10):
    """
    Added max_attempts limit to prevent infinite convergence loops
    """
    sequence = {}
    hpc_freq = []
    for read_id, read_seq in x:
        sequence[read_id] = read_seq
        hpc_freq.append((compress_seq(read_seq), [read_id, ]))

    res = iter_cluster_sequence(circ_id, hpc_freq, sequence)

    # If consensus - keep as 10 attempts
    for attempt in range(max_attempts):
        n_res = cluster_sequence(res, sequence)
        if len(n_res) == len(res):
            break
        res = n_res
    else:
        # Changed to debug level to reduce log spam
        LOGGER.debug('Sequence not consensus for circRNA: {} after {} attempts'.format(circ_id, max_attempts))
    return res


def iter_cluster_sequence(circ_id, hpc_freq, sequence, max_attempts=10):
    """
    Added max_attempts limit
    """
    if len(hpc_freq) <= 50:
        return cluster_sequence(hpc_freq, sequence)

    res = []
    for tmp in grouper(hpc_freq, 50):
        chunk = [i for i in tmp if i is not None]
        res = cluster_sequence(chunk + res, sequence)

        for _ in range(max_attempts):
            n_res = cluster_sequence(res, sequence)
            if len(n_res) == len(res):
                break
            res = n_res
        else:
            # Changed to debug level to reduce log spam
            LOGGER.debug('Sequence not consensus for circRNA: {} in iter after {} attempts'.format(circ_id, max_attempts))
    return res


def cluster_sequence(hpc_freq, sequence):
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from spoa import poa

    if len(hpc_freq) == 1:
        return hpc_freq

    # Calculate distance between segments
    dist = np.zeros((len(hpc_freq), len(hpc_freq)))
    for i in range(len(hpc_freq)):
        for j in range(len(hpc_freq)):
            if i > j:
                continue
            dist[i][j] = distance(hpc_freq[i][0], hpc_freq[j][0]) / max(len(hpc_freq[i][0]), len(hpc_freq[j][0]))
    dist = dist + dist.T

    # Hierarchical Cluster
    if dist.sum() != 0:
        z = leaves_list(linkage(squareform(dist), "ward", optimal_ordering=True))
    else:
        z = list(range(len(hpc_freq)))

    clusters = [[z[0], ]]
    for i, j in pairwise(z):
        if i > j:
            if dist[j][i] < 0.3:
                clusters[-1].append(j)
            else:
                clusters.append([j, ])
        else:
            if dist[i][j] < 0.3:
                clusters[-1].append(j)
            else:
                clusters.append([j, ])

    ccs_seq = []
    for cluster in clusters:
        if len(cluster) == 1:
            ccs_seq.append((hpc_freq[cluster[0]]))
            continue

        cluster_reads = flatten([hpc_freq[i][1] for i in cluster])
        cluster_seq = [sequence[i] for i in cluster_reads]

        # Generate consensus sequence
        ccs, _ = poa(cluster_seq, 2, False, 10, -4, -8, -2, -24, -1)
        ccs_seq.append((ccs, cluster_reads))
    return ccs_seq


def find_intron_signal(contig, starts, ends):
    if env.SS_INDEX is not None and contig in env.SS_INDEX:
        return None

    for strand in ['+', '-']:
        tmp_us_sites = []
        for st in range(min(starts)-10, max(starts)+10):
            us_pos = st
            if us_pos not in env.SS_INDEX[contig]:
                continue
            if strand not in env.SS_INDEX[contig][us_pos]:
                continue
            if 'end' not in env.SS_INDEX[contig][us_pos][strand]:
                continue
            tmp_us_sites.append(st)

        tmp_ds_sites = []
        for en in range(min(ends)-10, max(ends)+10):
            ds_pos = en
            if ds_pos not in env.SS_INDEX[contig]:
                continue
            if strand not in env.SS_INDEX[contig][ds_pos]:
                continue
            if 'start' not in env.SS_INDEX[contig][ds_pos][strand]:
                continue
            tmp_ds_sites.append(en)

        if len(tmp_us_sites) == 0 and len(tmp_ds_sites) == 0:
            continue
        elif len(tmp_us_sites) > 0 and len(tmp_ds_sites) > 0:
            pass
        elif len(tmp_us_sites) > 0:
            pass
        elif len(tmp_ds_sites) > 0:
            pass
        else:
            continue


def recursive_splice_site(scores, ctg, strand):
    for st, en, scr in scores:
        if strand == '+' and (env.GENOME.seq(ctg, st-2, st) == 'AG' and env.GENOME.seq(ctg, st, st+2) == 'GT'):
            return st, en, scr
        if strand == '-' and (env.GENOME.seq(ctg, en, en+2) == 'CT' and env.GENOME.seq(ctg, en-2, en) == 'CA'):
            return st, en, scr
    return None, None, None


def curate_cirexons(circ, cluster):
    isoforms = {}
    starts = []
    ends = []
    for read in cluster:
        if read.cirexon == 'NA':
            continue

        exons = parse_cirexons(circ, read)
        if len(exons) == 0:
            continue

        for exon, exon_type in exons:
            if exon_type != '*-':
                starts.append(exon.start)
            if exon_type != '-*':
                ends.append(exon.end)

        if read.type == 'partial':
            continue
        isoforms[read.read_id] = [i[0] for i in exons]

    if len(isoforms) == 0:
        return None

    # Cluster junction sites
    tmp_starts = cluster_bins(starts, dis=10)
    tmp_ends = cluster_bins(ends, dis=10)

    convert_st = {}
    for tmp_st in tmp_starts:
        # circRNA start
        if circ.start in tmp_st:
            for i in tmp_st:
                convert_st[i] = circ.start

        # Canonical sites
        aval_st = []
        for i in set(tmp_st):
            i_ss = env.GENOME.seq(circ.contig, i-3, i-1)
            if circ.strand == '+' and i_ss == 'AG':
                aval_st.append(i)
            elif circ.strand == '-' and revcomp(i_ss) == 'GT':
                aval_st.append(i)
            else:
                pass

        tmp_st_counter = Counter(tmp_st)
        if aval_st:
            final_st = sorted(aval_st, key=lambda x: tmp_st_counter[x], reverse=True)[0]
        else:
            final_st = tmp_st_counter.most_common(n=1)[0][0]
        for i in tmp_st:
            convert_st[i] = final_st

    convert_en = {}
    for tmp_en in tmp_ends:
        # circRNA start
        if circ.end in tmp_en:
            for i in tmp_en:
                convert_en[i] = circ.end

        # Canonical sites
        aval_en = []
        for i in set(tmp_en):
            i_ss = env.GENOME.seq(circ.contig, i, i + 2)
            if circ.strand == '+' and i_ss == 'GT':
                aval_en.append(i)
            elif circ.strand == '-' and revcomp(i_ss) == 'AG':
                aval_en.append(i)
            else:
                pass

        tmp_en_counter = Counter(tmp_en)
        if aval_en:
            final_en = sorted(aval_en, key=lambda x: tmp_en_counter[x], reverse=True)[0]
        else:
            final_en = tmp_en_counter.most_common(n=1)[0][0]
        for i in tmp_en:
            convert_en[i] = final_en

    curated_exons = {}
    for read_id, exons in isoforms.items():
        tmp_exons = [Exon(convert_st[exon.start], convert_en[exon.end]) for exon in exons]

        while tmp_exons[0].end <= circ.start:
            tmp_exons = tmp_exons[1:]
            if len(tmp_exons) == 0:
                break
        if len(tmp_exons) == 0:
            continue

        while tmp_exons[-1].start >= circ.end:
            tmp_exons = tmp_exons[:-1]
            if len(tmp_exons) == 0:
                break
        if len(tmp_exons) == 0:
            continue

        tmp_exons = merge_cirexons(tmp_exons)
        if tmp_exons[0].start <= circ.start + 15 and tmp_exons[-1].end >= circ.end - 15:
            tmp_exons[0].start = circ.start
            tmp_exons[-1].end = circ.end
        else:
            continue

        curated_exons[read_id] = tmp_exons

    return curated_exons


def merge_cirexons(exons):
    if len(exons) == 1:
        return exons

    last_exon = exons[0]
    merged_exons = []
    for exon in exons[1:]:
        if exon.start <= last_exon.end + 10:
            last_exon = Exon(last_exon.start, exon.end)
        else:
            merged_exons.append(last_exon)
            last_exon = exon
    merged_exons.append(last_exon)

    return merged_exons


def curate_isoform(circ, curated_exons, cluster_res):
    final_isoforms = {}
    for tmp_seq, tmp_ids in cluster_res:
        tmp_isoform, tmp_len = merge_isoforms(circ, curated_exons, tmp_seq, tmp_ids)
        if tmp_isoform is None:
            continue

        if tmp_isoform in final_isoforms:
            final_isoforms[tmp_isoform][1] += tmp_ids
        else:
            final_isoforms[tmp_isoform] = [tmp_len, tmp_ids]
    if len(final_isoforms) == 0:
        return None, None, None

    total_cnt = sum([len(i[1]) for i in final_isoforms.values()])
    ret = sorted(list(final_isoforms),
                 key=lambda x: (len(final_isoforms[x][1]), final_isoforms[x][0]),
                 reverse=True)
    major_len = final_isoforms[ret[0]][0]
    major_isoforms = [i for i in ret if len(final_isoforms[i][1]) >= 0.1 * total_cnt]
    major_reads = [final_isoforms[i][1] for i in major_isoforms]
    return major_isoforms, major_reads, major_len


def merge_isoforms(circ, curated_exons, seq, ids):
    from libs.striped_smith_waterman.ssw_wrap import Aligner
    aligner = Aligner(seq, match=10, mismatch=4, gap_open=8, gap_extend=2)

    tmp = [i for i in ids if i in curated_exons]
    exons = sorted(set([str(j) for i in tmp for j in curated_exons[i]]))
    if len(exons) == 0:
        return None, None

    exons = ['st', ] + exons + ['en']
    edges = np.zeros([len(exons), len(exons)])
    for i in tmp:
        tmp_exons = [str(j) for j in curated_exons[i]]
        edges[exons.index('st')][exons.index(tmp_exons[0])] += 1
        edges[exons.index(tmp_exons[-1])][exons.index('en')] += 1

        for l_exon, n_exon in pairwise(tmp_exons):
            edges[exons.index(l_exon)][exons.index(n_exon)] += 1

    cand_st, cand_en = np.where(edges == np.amax(edges))

    cand_score = [exon_score(circ, aligner, exons[i], exons[j]) for i, j in zip(cand_st, cand_en)]

    cand_idx = np.where(cand_score == np.amax(cand_score))[0][0]

    max_flow = []
    max_flow += iter_flow(circ, aligner, exons, edges, cand_st[cand_idx], -1)
    max_flow += iter_flow(circ, aligner, exons, edges, cand_en[cand_idx], 1)

    isoform = [exons[i] for i in max_flow]
    isoform_id = ','.join([i for i in isoform[1:-1]])
    isoform_len = sum([int(i.split('-')[1]) - int(i.split('-')[0]) + 1 for i in isoform[1:-1]])
    return isoform_id, isoform_len


def iter_flow(circ, aligner, exons, edges, coord, direction=-1):
    if coord == 0 or coord == edges.shape[0] - 1:
        return [coord, ]

    if direction == -1:
        max_l = np.where(edges[:, coord] == np.amax(edges[:, coord]))[0]
        max_score = [exon_score(circ, aligner, exons[i], exons[coord]) for i in max_l]
        max_idx = max_l[np.where(max_score == np.amax(max_score))[0][0]]
        return iter_flow(circ, aligner, exons, edges, max_idx, direction) + [coord, ]
    else:
        max_n = np.where(edges[coord] == np.amax(edges[coord]))[0]
        max_score = [exon_score(circ, aligner, exons[coord], exons[i]) for i in max_n]
        max_idx = max_n[np.where(max_score == np.amax(max_score))[0][0]]
        return [coord, ] + iter_flow(circ, aligner, exons, edges, max_idx, direction)


def exon_score(circ, aligner, l_exon, n_exon):
    query_seq = ''
    if l_exon != 'st':
        l_st, l_en = l_exon.split('-')
        tmp = env.GENOME.seq(circ.contig, int(l_st)-1, int(l_en))
        query_seq += tmp
    if n_exon != 'en':
        n_st, n_en = n_exon.split('-')
        tmp = env.GENOME.seq(circ.contig, int(n_st), int(n_en))
        query_seq += tmp
    if circ.strand == '-':
        query_seq = revcomp(query_seq)

    alignment = aligner.align(query_seq)
    return alignment.ref_end - alignment.ref_begin


def parse_cirexons(circ, read):
    exon_str = read.cirexon.split(',')
    exons = []
    for x in exon_str:
        st, en = x.split('|')[0].split('-')
        exons.append([Exon(st, en), x.split('|')[1]])
    return exons


def cluster_bins(pos, dis=10):
    clustered = []
    last_i = None
    for i in sorted(pos):
        if last_i is None:
            last_i = [i, ]
            continue
        if i > last_i[-1] + dis:
            clustered.append(last_i)
            last_i = [i, ]
        else:
            last_i.append(i)
    clustered.append(last_i)
    return clustered


def intron_splice_sites(contig, start, end, strand):
    us = env.GENOME.seq(contig, start, start + 2)
    ds = env.GENOME.seq(contig, end - 3, end - 1)
    if strand == '-':
        us, ds = revcomp(ds), revcomp(us)
    return signal_weight(us, ds)


def signal_weight(us, ds):
    if us == 'GT' and ds == 'AG':
        return 0
    else:
        return 1


def check_isoforms(circ, isoforms):
    concordance = []
    for iso_str in isoforms:
        exons = iso_str.split(',')
        if len(exons) == 1:
            concordance.append(True)
            continue

        introns = []
        for l_str, n_str in pairwise(exons):
            l_st, l_en = l_str.split('-')
            n_st, n_en = n_str.split('-')
            l_ss = env.GENOME.seq(circ.contig, int(l_en), int(l_en) + 2)
            n_ss = env.GENOME.seq(circ.contig, int(n_st) - 3, int(n_st) - 1)
            if circ.strand == '+' and l_ss == 'GT' and n_ss == 'AG':
                introns.append(1)
            elif circ.strand == '-' and revcomp(n_ss) == 'GT' and revcomp(l_ss) == 'AG':
                introns.append(1)
            else:
                introns.append(0)
        concordance.append(sum(introns) == len(introns))

    return sum(concordance) > 0


def correct_reads(reads_cluster, ref_fasta, gtf_index, intron_index, ss_index, threads, 
                  checkpoint_file=None, max_cluster=100):
    """
    Optimized version with checkpointing, better parallelization, and progress tracking
    """
    from multiprocessing import Pool
    from CIRI_long.logger import ProgressBar
    
    # Load reference genome
    genome = Fasta(ref_fasta)

    corrected_reads = []
    circ_num = defaultdict(int)
    
    # Load checkpoint if exists
    start_idx = 0
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                start_idx = checkpoint_data['index']
                corrected_reads = checkpoint_data['corrected_reads']
                circ_num = checkpoint_data['circ_num']
            LOGGER.info(f'Resuming from checkpoint: {start_idx} jobs completed')
        except Exception as e:
            LOGGER.warn(f'Failed to load checkpoint: {e}. Starting from beginning.')
            start_idx = 0
    
    # Prepare jobs with dynamic chunk sizing
    total_reads = sum(len(cluster) for cluster in reads_cluster)
    avg_cluster_size = total_reads / len(reads_cluster) if reads_cluster else 0
    
    # Adjust chunk size based on cluster complexity
    if avg_cluster_size > 500:
        chunk_size = 50  # Smaller chunks for complex data
    elif avg_cluster_size > 200:
        chunk_size = 100
    else:
        chunk_size = 250  # Original size for simple data
    
    LOGGER.info(f'Processing {len(reads_cluster)} clusters (avg size: {avg_cluster_size:.1f}) with chunk size {chunk_size}')
    
    jobs = []
    pool = Pool(threads, env.initializer, 
                (None, genome.contig_len, genome, gtf_index, intron_index, ss_index,))

    # Create job chunks
    all_chunks = list(grouper(reads_cluster, chunk_size))
    
    # Skip already processed jobs
    for chunk in all_chunks[start_idx:]:
        jobs.append(pool.apply_async(correct_chunk, (chunk, max_cluster, 60)))  # 60s timeout
    
    pool.close()

    prog = ProgressBar()
    total_jobs = len(all_chunks)
    completed_jobs = start_idx
    prog.update(100 * completed_jobs // total_jobs if total_jobs > 0 else 0)
    
    LOGGER.info(f'Processing {len(jobs)} remaining job chunks ({completed_jobs}/{total_jobs} already complete)')
    
    # Process results with timeout handling
    start_time = time.time()
    last_update = start_time
    
    for job_idx, job in enumerate(jobs):
        try:
            # Get result with timeout
            tmp_cluster, tmp_num = job.get(timeout=3600)  # 1 hour timeout per chunk
            
            corrected_reads += tmp_cluster
            for i in tmp_num:
                circ_num[i] += tmp_num[i]
            
            completed_jobs += 1
            
            # Update progress
            current_time = time.time()
            if current_time - last_update > 5:  # Update every 5 seconds
                elapsed = current_time - start_time
                rate = (completed_jobs - start_idx) / elapsed if elapsed > 0 else 0
                remaining = (total_jobs - completed_jobs) / rate if rate > 0 else 0
                LOGGER.info(f'Progress: {completed_jobs}/{total_jobs} ({100*completed_jobs//total_jobs}%) - '
                           f'Rate: {rate:.2f} jobs/sec - ETA: {remaining/60:.1f} min')
                last_update = current_time
            
            prog.update(100 * completed_jobs // total_jobs)
            
            # Save checkpoint every 10% or every 50 jobs
            if checkpoint_file and (completed_jobs % max(1, total_jobs // 10) == 0 or 
                                   (completed_jobs - start_idx) % 50 == 0):
                try:
                    checkpoint_data = {
                        'index': completed_jobs,
                        'corrected_reads': corrected_reads,
                        'circ_num': circ_num
                    }
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                    LOGGER.info(f'Checkpoint saved at job {completed_jobs}')
                except Exception as e:
                    LOGGER.warn(f'Failed to save checkpoint: {e}')
                    
        except Exception as e:
            LOGGER.error(f'Job {job_idx + start_idx} failed: {e}')
            completed_jobs += 1
            continue
    
    pool.join()
    prog.update(100)
    
    # Save final checkpoint
    if checkpoint_file:
        try:
            checkpoint_data = {
                'index': completed_jobs,
                'corrected_reads': corrected_reads,
                'circ_num': circ_num
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            LOGGER.info('Final checkpoint saved')
        except Exception as e:
            LOGGER.warn(f'Failed to save final checkpoint: {e}')
    
    # Summary statistics
    total_time = time.time() - start_time
    LOGGER.info(f'Processing completed in {total_time/60:.1f} minutes')
    if circ_num:
        LOGGER.info(f'Results: {sum(circ_num.values())} total, breakdown: {dict(circ_num)}')

    return circ_num, corrected_reads


def circ_pos(x):
    ctg, pos = x.split(':')
    st, en = pos.split('-')
    return ctg, int(st), int(en)


def by_circ(x):
    ctg, pos = x.split(':')
    if ctg.startswith('chr'):
        ctg = ctg.lstrip('chr')

    try:
        idx = '{:02d}'.format(int(ctg))
    except ValueError as e:
        if ctg in ['X', 'x', 'Y', 'y']:
            idx = 'a'
        elif ctg in ['M', 'm']:
            idx = 'b'
        else:
            idx = 'c'

    st, en = pos.split('-')

    return idx, ctg, int(st), int(en)


def by_isoform(x):
    circ_id, iso_id = x.split('|')
    idx, ctg, st, en = by_circ(circ_id)
    return idx, ctg, st, en, iso_id


def cal_exp_mtx(cand_reads, corrected_reads, ref_fasta, gtf_idx, out_dir, prefix):
    from collections import Counter

    genome = Fasta(ref_fasta)

    circ_reads = defaultdict(list)
    isoform_reads = defaultdict(dict)
    circ_info = {}
    reads_df = []

    for reads, tmp_iso_reads, seqs, circ_id, strand, ss_id, us_free, ds_free, circ_len, isoforms in corrected_reads:
        # circRNA information
        ctg, st, en = circ_pos(circ_id)

        # Remove circRNAs shorter than 30 bp
        if en - st < 30:
            continue

        field = circ_attr(gtf_idx, ctg, st, en, strand)

        tmp_attr = 'circ_id "{}"; splice_site "{}"; equivalent_seq "{}"; circ_type "{}"; circ_len "{}";'.format(
            circ_id,
            ss_id,
            equivalent_seq(genome, ctg, st, en, strand),
            field['circ_type'] if field else 'Unknown',
            circ_len,
        )
        if isoforms:
            tmp_attr += ' isoform "{}";'.format('|'.join(isoforms))

        for key in 'gene_id', 'gene_name', 'gene_type':
            if key in field:
                tmp_attr += ' {} "{}";'.format(key, field[key])
        circ_info[circ_id] = [ctg, 'CIRI-long', 'circRNA', st, en, len(reads), strand, '.', tmp_attr, ]

        # Expression levels
        circ_reads[circ_id] += reads
        for i, j in zip(isoforms, tmp_iso_reads):
            isoform_reads[circ_id][i] = isoform_reads[circ_id].setdefault(i, []) + j

        # Corrected reads
        for read_id in reads:
            read = cand_reads[read_id]
            tmp = [read_id, circ_id, read.circ_id, read.strand, read.cirexon,
                   read.ss, read.clip, read.segments, read.sample, read.type]
            reads_df.append(tmp)

    # circular reads
    reads_df = pd.DataFrame(
        reads_df, columns=['read_id', 'circ_id', 'tmp_id', 'strand', 'cirexons',
                           'signal', 'alignment', 'segments', 'sample', 'type'])
    reads_df.to_csv('{}/{}.reads'.format(out_dir, prefix), sep="\t", index=False)

    # circRNA information
    sorted_circ = sorted(list(circ_info), key=by_circ)
    with open('{}/{}.info'.format(out_dir, prefix), 'w') as out:
        for circ_id in sorted_circ:
            tmp_line = circ_info[circ_id]
            out.write('\t'.join([str(x) for x in tmp_line]) + '\n')

    # Expression level
    exp_df = {}
    for circ_id, reads in circ_reads.items():
        exp_df[circ_id] = Counter([cand_reads[i].sample for i in reads])
    exp_df = pd.DataFrame.from_dict(exp_df).transpose().fillna(0).reindex(sorted_circ)
    exp_df.to_csv('{}/{}.expression'.format(out_dir, prefix), sep="\t", index_label='circ_ID')

    isoform_df = {}
    for circ_id in isoform_reads:
        tmp_total = []
        for _, reads in isoform_reads[circ_id].items():
            tmp_total += [cand_reads[i].sample for i in reads]
        tmp_total = Counter(tmp_total)
        for iso_id, reads in isoform_reads[circ_id].items():
            tmp_counter = Counter([cand_reads[i].sample for i in reads])
            tmp_iso = {i: j/tmp_total[i] for i, j in tmp_counter.items()}
            isoform_df['{}|{}'.format(circ_id, iso_id)] = tmp_iso
    sorted_iso = sorted(list(isoform_df), key=by_isoform)
    isoform_df = pd.DataFrame.from_dict(isoform_df).transpose().fillna(0).reindex(sorted_iso)
    isoform_df.to_csv('{}/{}.isoforms'.format(out_dir, prefix), sep="\t", index_label='isoform_ID')
    return len(sorted_circ), len(sorted_iso)


def equivalent_seq(genome, contig, start, end, strand):
    if strand is None:
        return 'Unknown'

    ds_seq = ''
    for i in range(100):
        if end + i > genome.contig_len[contig]:
            break
        if genome.seq(contig, start-1, start-1 + i) == genome.seq(contig, end, end + i):
            ds_seq = genome.seq(contig, start-1, start-1 + i)
        else:
            break

    us_seq = ''
    for j in range(100):
        if start - j < 0:
            break
        if genome.seq(contig, start-1 - j, start-1) == genome.seq(contig, end - j, end):
            us_seq = genome.seq(contig, start-1 - j, start-1)
        else:
            break

    tmp = us_seq + ds_seq
    if strand == '+':
        return tmp
    else:
        return revcomp(tmp)


def circ_attr(gtf_index, ctg, start, end, strand):
    """
    annotate circRNA information
    """
    if gtf_index is None or ctg not in gtf_index:
        return {}
    start_div, end_div = start // 500, end // 500

    host_gene = {}
    start_element = defaultdict(list)
    end_element = defaultdict(list)

    for x in range(start_div, end_div + 1):
        if x not in gtf_index[ctg]:
            continue
        for element in gtf_index[ctg][x]:
            # start site
            if element.start <= start <= element.end and (element.strand == strand or strand is None):
                start_element[element.type].append(element)
            # end site
            if element.start <= end <= element.end and (element.strand == strand or strand is None):
                end_element[element.type].append(element)
            if element.end < start or end < element.start:
                continue
            if element.attr['gene_id'] not in host_gene:
                host_gene[element.attr['gene_id']] = element

    circ_type = {}
    forward_host_gene = []
    antisense_host_gene = []

    if len(host_gene) > 0:
        for gene_id in host_gene:
            if strand == 'None' or host_gene[gene_id].strand == strand:
                forward_host_gene.append(host_gene[gene_id])
                if 'exon' in start_element and 'exon' in end_element:
                    circ_type['exon'] = 1
                else:
                    circ_type['intron'] = 1
            else:
                antisense_host_gene.append(host_gene[gene_id])
                circ_type['antisense'] = 1
    else:
        circ_type['intergenic'] = 1

    if len(forward_host_gene) > 1:
        circ_type['gene_intergenic'] = 1

    field = {}
    if 'exon' in circ_type:
        field['circ_type'] = 'exon'
    elif 'intron' in circ_type:
        field['circ_type'] = 'intron'
    elif 'antisense' in circ_type:
        field['circ_type'] = 'antisense'
    else:
        field['circ_type'] = 'intergenic'

    if len(forward_host_gene) == 1:
        if 'gene_id' in forward_host_gene[0].attr:
            field['gene_id'] = forward_host_gene[0].attr['gene_id']
        if 'gene_name' in forward_host_gene[0].attr:
            field['gene_name'] = forward_host_gene[0].attr['gene_name']
        if 'gene_type' in forward_host_gene[0].attr:
            field['gene_type'] = forward_host_gene[0].attr['gene_type']
        elif 'gene_biotype' in forward_host_gene[0].attr:
            field['gene_type'] = forward_host_gene[0].attr['gene_biotype']
        else:
            pass
    elif len(forward_host_gene) > 1:
        tmp_gene_id = []
        tmp_gene_name = []
        tmp_gene_type = []
        for x in forward_host_gene:
            if 'gene_id' in x.attr:
                tmp_gene_id.append(x.attr['gene_id'])
            if 'gene_name' in x.attr:
                tmp_gene_name.append(x.attr['gene_name'])
            if 'gene_type' in x.attr:
                tmp_gene_type.append(x.attr['gene_type'])
            elif 'gene_biotype' in x.attr:
                tmp_gene_type.append(x.attr['gene_biotype'])
            else:
                pass
        if len(tmp_gene_id) > 0:
            field['gene_id'] = ','.join(tmp_gene_id)
        if len(tmp_gene_name) > 0:
            field['gene_name'] = ','.join(tmp_gene_name)
        if len(tmp_gene_type) > 0:
            field['gene_type'] = ','.join(tmp_gene_type)
    elif field['circ_type'] == 'antisense' and len(antisense_host_gene) > 0:
        tmp_gene_id = []
        tmp_gene_name = []
        tmp_gene_type = []
        for x in antisense_host_gene:
            if 'gene_id' in x.attr:
                tmp_gene_id.append(x.attr['gene_id'])
            if 'gene_name' in x.attr:
                tmp_gene_name.append(x.attr['gene_name'])
            if 'gene_type' in x.attr:
                tmp_gene_type.append(x.attr['gene_type'])
            elif 'gene_biotype' in x.attr:
                tmp_gene_type.append(x.attr['gene_biotype'])
            else:
                pass
        if len(tmp_gene_id) > 0:
            field['gene_id'] = ','.join(tmp_gene_id)
        if len(tmp_gene_name) > 0:
            field['gene_name'] = ','.join(tmp_gene_name)
        if len(tmp_gene_type) > 0:
            field['gene_type'] = ','.join(tmp_gene_type)
    else:
        pass

    return field
