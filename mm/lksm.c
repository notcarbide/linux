// SPDX-License-Identifier: GPL-2.0-only
/*
 * Lightweight KSM.
 *
 * This code provides lightweight version of KSM.
 *
 * Copyright (C) 2020 Samsung Electronics Co., Ltd.
 * Author: Sung-hun Kim (sfoon.kim@samsung.com)
 */

/*
 * Memory merging support.
 *
 * This code enables dynamic sharing of identical pages found in different
 * memory areas, even if they are not shared by fork()
 *
 * Copyright (C) 2008-2009 Red Hat, Inc.
 * Authors:
 *	Izik Eidus
 *	Andrea Arcangeli
 *	Chris Wright
 *	Hugh Dickins
 */

#include <linux/errno.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/mman.h>
#include <linux/sched.h>
#include <linux/sched/mm.h>
#include <linux/sched/coredump.h>
#include <linux/rwsem.h>
#include <linux/pagemap.h>
#include <linux/rmap.h>
#include <linux/spinlock.h>
#include <linux/xxhash.h>
#include <linux/delay.h>
#include <linux/kthread.h>
#include <linux/wait.h>
#include <linux/slab.h>
#include <linux/rbtree.h>
#include <linux/memory.h>
#include <linux/mmu_notifier.h>
#include <linux/swap.h>
#include <linux/ksm.h>
#include <linux/hashtable.h>
#include <linux/freezer.h>
#include <linux/oom.h>
#include <linux/numa.h>

#include <asm/tlbflush.h>
#include "internal.h"

#ifdef CONFIG_NUMA
#define NUMA(x)		(x)
#define DO_NUMA(x)	do { (x); } while (0)
#else
#define NUMA(x)		(0)
#define DO_NUMA(x)	do { } while (0)
#endif

#define ksm_debug(fmt, ...) \
	pr_debug("[ksm:%s:%d] " fmt "\n", __func__, __LINE__, ##__VA_ARGS__)
#define ksm_err(fmt, ...) \
	pr_err("[ksm:%s:%d] " fmt "\n", __func__, __LINE__, ##__VA_ARGS__)

/**
 * DOC: Overview
 *
 * A few notes about the KSM scanning process,
 * to make it easier to understand the data structures below:
 *
 * In order to reduce excessive scanning, KSM sorts the memory pages by their
 * contents into a data structure that holds pointers to the pages' locations.
 *
 * Since the contents of the pages may change at any moment, KSM cannot just
 * insert the pages into a normal sorted tree and expect it to find anything.
 * Therefore KSM uses two data structures - the stable and the unstable tree.
 *
 * The stable tree holds pointers to all the merged pages (ksm pages), sorted
 * by their contents.  Because each such page is write-protected, searching on
 * this tree is fully assured to be working (except when pages are unmapped),
 * and therefore this tree is called the stable tree.
 *
 * The stable tree node includes information required for reverse
 * mapping from a KSM page to virtual addresses that map this page.
 *
 * In order to avoid large latencies of the rmap walks on KSM pages,
 * KSM maintains two types of nodes in the stable tree:
 *
 * * the regular nodes that keep the reverse mapping structures in a
 *   linked list
 * * the "chains" that link nodes ("dups") that represent the same
 *   write protected memory content, but each "dup" corresponds to a
 *   different KSM page copy of that content
 *
 * Internally, the regular nodes, "dups" and "chains" are represented
 * using the same :c:type:`struct stable_node` structure.
 *
 * In addition to the stable tree, KSM uses a second data structure called the
 * unstable tree: this tree holds pointers to pages which have been found to
 * be "unchanged for a period of time".  The unstable tree sorts these pages
 * by their contents, but since they are not write-protected, KSM cannot rely
 * upon the unstable tree to work correctly - the unstable tree is liable to
 * be corrupted as its contents are modified, and so it is called unstable.
 *
 * KSM solves this problem by several techniques:
 *
 * 1) The unstable tree is flushed every time KSM completes scanning all
 *    memory areas, and then the tree is rebuilt again from the beginning.
 * 2) KSM will only insert into the unstable tree, pages whose hash value
 *    has not changed since the previous scan of all memory areas.
 * 3) The unstable tree is a RedBlack Tree - so its balancing is based on the
 *    colors of the nodes and not on their contents, assuring that even when
 *    the tree gets "corrupted" it won't get out of balance, so scanning time
 *    remains the same (also, searching and inserting nodes in an rbtree uses
 *    the same algorithm, so we have no overhead when we flush and rebuild).
 * 4) KSM never flushes the stable tree, which means that even if it were to
 *    take 10 attempts to find a page in the unstable tree, once it is found,
 *    it is secured in the stable tree.  (When we scan a new page, we first
 *    compare it against the stable tree, and then against the unstable tree.)
 *
 * If the merge_across_nodes tunable is unset, then KSM maintains multiple
 * stable trees and multiple unstable trees: one of each for each NUMA node.
 */

/*
 * A few notes about lightweight KSM.
 *
 * A smart crawler leverages semantics of tasks in Tizen.
 * When the application goes to background, it is attached to freezer
 * task group. LKSM crawler hooks this event and adds a "frozen task"
 * to candidate list for scanning.
 *
 */

/* merge window size */
#define MERGE_WIN 3

/**
 * struct mm_slot - ksm information per mm that is being scanned
 * @link: link to the mm_slots hash list
 * @mm_list: link into the mm_slots list, rooted in ksm_mm_head
 * @rmap_list: head for this mm_slot's singly-linked list of rmap_items
 * @mm: the mm that this information is valid for
 *
 * extension - added for LKSM
 * @state: state of mm_slot (frozen, listed, scanned, newcomer)
 * @merge_idx: merge window index to store the number of currently merged pages
 * @nr_merged_win: merge window to keep recent three numbers
 * @nr_merged: sum of nr_merged_win, used to maintain vips_list (ordered list)
 * @ordered_list: list ordered by nr_merged
 * @scanning_size: number of anonymous pages in mm_struct
 * @fault_cnt: last read count of page fault (minor + major)
 * @elapsed: elapsed scanning time
 * @nr_scans: number of scanning pages (can be different with scanning_size)
 */
struct mm_slot {
	struct hlist_node link;
	struct list_head mm_list;
	struct list_head scan_list;
	struct rmap_item *rmap_list;
	struct mm_struct *mm;

	short state;

	short merge_idx;
	int nr_merged_win[MERGE_WIN];
	int nr_merged;
	struct rb_node ordered_list;

	unsigned long scanning_size; /* in number of pages */
	unsigned long fault_cnt;
	unsigned long elapsed;
	int nr_scans;

#ifdef CONFIG_LKSM_FILTER
	/* used for releasing lksm_region */
	struct list_head ref_list;
	int nr_regions;
#endif

};

/*
 * scanning mode of LKSM:
 * LKSM_SCAN_PARTIAL: perform deduplication on subset of processes
 * LKSM_SCAN_FULL: perform deduplication on full set of processes
 */
enum lksm_scan_mode {
	LKSM_SCAN_NONE,
	LKSM_SCAN_PARTIAL,
	LKSM_SCAN_FULL,
};

/**
 * struct ksm_scan - cursor for scanning
 * @address: the next address inside that to be scanned
 * @rmap_list: link to the next rmap to be scanned in the rmap_list
 * @mm_slot: the current mm_slot we are scanning
 * @remove_mm_list: temporary list for batching flush of removed slots
 * @nr_scannable: the number of remaining unscanned scannable slots
 * @nr_frozen: the number of remaining unscanned frozen slots
 * @scan_round: scanning round (partial + full)
 * @nr_full_scan: the number of full scanning
 * @scan_mode: coverage of current scanning
 *
 * There is only the one ksm_scan instance of this cursor structure.
 */
struct ksm_scan {
	unsigned long address;
	struct rmap_item **rmap_list;

	struct mm_slot *mm_slot;
	struct list_head remove_mm_list;

	/* statistics of scanning targets */
	atomic_t nr_scannable;
	atomic_t nr_frozen;

	unsigned long scan_round;
	unsigned long nr_full_scan;

	enum lksm_scan_mode scan_mode;

#ifdef CONFIG_LKSM_FILTER
	struct lksm_region *region;
	unsigned long vma_base_addr;
	struct vm_area_struct *cached_vma;
#endif /* CONFIG_LKSM_FILTER */
};

/**
 * struct stable_node - node of the stable rbtree
 * @node: rb node of this ksm page in the stable tree
 * @head: (overlaying parent) &migrate_nodes indicates temporarily on that list
 * @hlist_dup: linked into the stable_node->hlist with a stable_node chain
 * @list: linked into migrate_nodes, pending placement in the proper node tree
 * @hlist: hlist head of rmap_items using this ksm page
 * @kpfn: page frame number of this ksm page (perhaps temporarily on wrong nid)
 * @chain_prune_time: time of the last full garbage collection
 * @rmap_hlist_len: number of rmap_item entries in hlist or STABLE_NODE_CHAIN
 * @nid: NUMA node id of stable tree in which linked (may not match kpfn)
 */
struct stable_node {
	union {
		struct rb_node node;	/* when node of stable tree */
		struct {		/* when listed for migration */
			struct list_head *head;
			struct {
				struct hlist_node hlist_dup;
				struct list_head list;
			};
		};
	};
	struct hlist_head hlist;
	union {
		unsigned long kpfn;
		unsigned long chain_prune_time;
	};
	/*
	 * STABLE_NODE_CHAIN can be any negative number in
	 * rmap_hlist_len negative range, but better not -1 to be able
	 * to reliably detect underflows.
	 */
#define STABLE_NODE_CHAIN -1024
	int rmap_hlist_len;
#ifdef CONFIG_NUMA
	int nid;
#endif
};

/**
 * struct rmap_item - reverse mapping item for virtual addresses
 * @rmap_list: next rmap_item in mm_slot's singly-linked rmap_list
 * @anon_vma: pointer to anon_vma for this mm,address, when in stable tree
 * @nid: NUMA node id of unstable tree in which linked (may not match page)
 * @region: pointer to the mapped region (LKSM feature)
 * @mm: the memory structure this rmap_item is pointing into
 * @address: the virtual address this rmap_item tracks (+ flags in low bits)
 * @oldchecksum: previous checksum of the page at that virtual address
 * @node: rb node of this rmap_item in the unstable tree
 * @head: pointer to stable_node heading this list in the stable tree
 * @base_addr: used for calculating offset of the address (LKSM feature)
 * @hlist: link into hlist of rmap_items hanging off that stable_node
 */
struct rmap_item {
	struct rmap_item *rmap_list;
	union {
		struct anon_vma *anon_vma;	/* when stable */
#ifdef CONFIG_NUMA
		int nid;		/* when node of unstable tree */
#endif
#ifdef CONFIG_LKSM_FILTER
		struct lksm_region *region; /* when unstable */
#endif
	};
	struct mm_struct *mm;
	unsigned long address;		/* + low bits used for flags below */
	unsigned int oldchecksum;	/* when unstable (LSB is a frozen bit) */
	union {
		struct rb_node node;	/* when node of unstable tree */
		struct {		/* when listed from stable tree */
#ifdef CONFIG_LKSM_FILTER
			union {
				struct stable_node *head;
				unsigned long base_addr; /* temporal storage for merge */
			};
#else
			struct stable_node *head;
#endif /* CONFIG_LKSM_FILTER */
			struct hlist_node hlist;
		};
	};
};

#define SEQNR_MASK	0x0ff	/* low bits of unstable tree scan_round */
#define UNSTABLE_FLAG	0x100	/* is a node of the unstable tree */
#define STABLE_FLAG	0x200	/* is listed from the stable tree */

/* The stable and unstable tree heads */
static struct rb_root one_stable_tree[1] = { RB_ROOT };
static struct rb_root one_unstable_tree[1] = { RB_ROOT };
static struct rb_root *root_stable_tree = one_stable_tree;
static struct rb_root *root_unstable_tree = one_unstable_tree;

#define LKSM_NODE_ID 0

/* Recently migrated nodes of stable tree, pending proper placement */
static LIST_HEAD(migrate_nodes);
#define STABLE_NODE_DUP_HEAD ((struct list_head *)&migrate_nodes.prev)

/* list for VIP processes */
static struct rb_root vips_list = RB_ROOT;
static int lksm_max_vips = 20;

#define MM_SLOTS_HASH_BITS 10
static DEFINE_HASHTABLE(mm_slots_hash, MM_SLOTS_HASH_BITS);
static DEFINE_HASHTABLE(task_slots_hash, MM_SLOTS_HASH_BITS);

/*
 * two list heads in LKSM:
 *  - ksm_mm_head: a head for traversing whole list of processes,
		not used for scanning itself
 *  - ksm_scan_head: a head for a list of currently scanning processes
 */
static struct mm_slot ksm_mm_head = {
	.mm_list = LIST_HEAD_INIT(ksm_mm_head.mm_list),
};

static struct mm_slot ksm_scan_head = {
	.scan_list = LIST_HEAD_INIT(ksm_scan_head.scan_list),
};

static struct ksm_scan ksm_scan = {
	.mm_slot = &ksm_scan_head,
};

static struct kmem_cache *rmap_item_cache;
static struct kmem_cache *stable_node_cache;
static struct kmem_cache *mm_slot_cache;
static struct kmem_cache *task_slot_cache;

/* The number of nodes in the stable tree */
static unsigned long ksm_pages_shared;

/* The number of page slots additionally sharing those nodes */
static unsigned long ksm_pages_sharing;

/* The number of nodes in the unstable tree */
static unsigned long ksm_pages_unshared;

/* The number of rmap_items in use: to calculate pages_volatile */
static unsigned long ksm_rmap_items;

/* The number of stable_node chains */
static unsigned long ksm_stable_node_chains;

/* The number of stable_node dups linked to the stable_node chains */
static unsigned long ksm_stable_node_dups;

/* Delay in pruning stale stable_node_dups in the stable_node_chains */
static unsigned int ksm_stable_node_chains_prune_millisecs = 2000;

/* Maximum number of page slots sharing a stable node */
static int ksm_max_page_sharing = 256;

/* Number of pages ksmd should scan in one batch */
static unsigned int ksm_thread_pages_to_scan = 100;

/* Milliseconds ksmd should sleep between batches */
static unsigned int ksm_thread_sleep_millisecs = 20;

/* Checksum of an empty (zeroed) page */
static unsigned int zero_checksum __read_mostly;

/* Processes tracked by KSM thread */
static unsigned int ksm_nr_added_process;

/* Whether to merge empty (zeroed) pages with actual zero pages */
static bool ksm_use_zero_pages __read_mostly;

/* An indicator for KSM scanning */
static atomic_t ksm_one_shot_scanning;

/* Boosting when the scanner performs partial scan */
static unsigned int lksm_boosted_pages_to_scan = 100;
static unsigned int lksm_default_pages_to_scan = 100;

#ifdef CONFIG_NUMA
/* Zeroed when merging across nodes is not allowed */
static unsigned int ksm_merge_across_nodes = 1;
static int ksm_nr_node_ids = 1;
#else
#define ksm_merge_across_nodes	1U
#define ksm_nr_node_ids		1
#endif

/*
 * Default policy for KSM_RUN_ONESHOT:
 * KSM performs both scannings only when the user requests it.
 * If scanning is ended, both crawler and scanner threads are blocked until
 * the next request is coming.
 */
#define KSM_RUN_STOP	0
#define KSM_RUN_MERGE	1
#define KSM_RUN_UNMERGE	2
#define KSM_RUN_OFFLINE	4
#define KSM_RUN_ONESHOT 8

static unsigned long ksm_run = KSM_RUN_STOP;
static atomic_t ksm_state; /* 0: in crawling 1: in scanning */

#define lksm_check_scan_state(ksm_state) (atomic_read(&ksm_state) == 1)
#define lksm_set_scan_state(ksm_state) (atomic_set(&ksm_state, 1))
#define lksm_clear_scan_state(ksm_state) (atomic_set(&ksm_state, 0))

struct task_slot {
	struct task_struct *task;
	int frozen;
	unsigned long inserted;
	struct list_head list;
	struct hlist_node hlist;
};

/*
 * Frozen state:
 * When a process stops running on forground (e.g., going to background),
 * the system daemon (e.g., resourced) puts it to cgroup_freezer.
 * Once a process joins into freezer cgroup, the system kernel does not count
 * it as a runnable process, and thus it cannot be scheduled on CPU.
 * So, I regard processes in freezer cgroup as a frozen state and that can be
 * good candidates of memory deduplication.
 *
 * LKSM provides a hook to catch the moment that the process is being frozen.
 * With the hook, ksm crawler can get candidate list for memory deduplication.
 * (see kernel/cgroup_freezer.c)
 */
#define FROZEN_BIT 0x01
#define LISTED_BIT 0x02

#define lksm_test_rmap_frozen(rmap_item) (rmap_item->oldchecksum & FROZEN_BIT)
#define lksm_set_rmap_frozen(rmap_item) (rmap_item->oldchecksum |= FROZEN_BIT)
#define lksm_clear_rmap_frozen(rmap_item) (rmap_item->oldchecksum &= ~FROZEN_BIT)
#define lksm_clear_checksum_frozen(checksum) (checksum &= ~FROZEN_BIT)

#define KSM_MM_FROZEN 0x01
#define KSM_MM_LISTED 0x02
#define KSM_MM_NEWCOMER 0x04
#define KSM_MM_SCANNED 0x08
#ifdef CONFIG_LKSM_FILTER
#define KSM_MM_PREPARED 0x10
#endif

#define lksm_test_mm_state(mm_slot, bit) (mm_slot->state & bit)
#define lksm_set_mm_state(mm_slot, bit) (mm_slot->state |= bit)
#define lksm_clear_mm_state(mm_slot, bit) (mm_slot->state &= ~bit)

#ifdef CONFIG_LKSM_FILTER
#define LKSM_REGION_HASH_BITS 10
static DEFINE_HASHTABLE(lksm_region_hash, LKSM_REGION_HASH_BITS);
spinlock_t lksm_region_lock;

/*
 * LKSM uses the filter when the region is scanned more than
 * LKSM_REGION_MATURE round
 */
#define LKSM_REGION_MATURE 5
#define lksm_region_mature(round, region) \
		((round - region->scan_round) > LKSM_REGION_MATURE)

enum lksm_region_type {
	LKSM_REGION_HEAP,
	LKSM_REGION_STACK,
	LKSM_REGION_FILE1, /* file mapped region: data section */
	LKSM_REGION_FILE2, /* file mapped region: bss section */
	LKSM_REGION_CONFLICT, /* conflicted regions: do not filtering */
	LKSM_REGION_UNKNOWN,
};

static const char * const region_type_str[] = {
	"heap",
	"stack",
	"file_data",
	"file_bss",
	"conflicted",
	"unknown",
};

/* sharing statistics for each region type */
static int region_share[LKSM_REGION_UNKNOWN + 1];

/*
 * lksm_region: A region represents a physical mapped area.
 * Each process can have its own instance of a region, namely vma.
 * Regions for not-a-file-mapped areas like heap and stack just have
 * abstract representations as symbols.
 *
 * LKSM leverages the region for offset-based filtering.
 * Each region has a filter which records offsets of addresses of
 * shared pages in the region.
 * If once a region is matured, LKSM uses the filter to skip scanning of
 * unsharable pages.
 *
 * @type: type of region, refer above enumeration
 * @len: length of filter (in the number of 64-bit variables)
 * @ino: inode number if the region is mapped to file
 * @merge_cnt: the number of merged pages in the region
 * @filter_cnt: the number of set bits in filter
 * @scan_round: the birth scan round of this region
 * @conflict: the count of size changed, clue for conflict
 * @refcount: if it reaches zero, the region will be freed
 * @hnode: hash node for finding region by ino
 * @next: data region can have a next (bss) region
 * @prev: reverse pointer to data region
 *
 * A few notes about bitmap filter variable:
 * LKSM uses bitmap filter for skipping scan of unsharable pages.
 * If a region is smaller than 256KB (<= 64 pages),
 * it can be covered by a bitmap stored in a 64-bit variable.
 * LKSM only allocates a bitmap array as a filter when the region is
 * larger than 256KB, otherwise it uses a 64-bit variable as a filter.
 *
 * @filter: when the region is bigger than 64 pages
 * @single_filter: when the region is smaller than or equal to 64 pages
 */
#define SINGLE_FILTER_LEN 1 /* a region can be covered by single variable */

struct lksm_region {
	enum lksm_region_type type;
	int ino;
	int merge_cnt;
	int filter_cnt;
	int scan_round;
	int conflict;
	unsigned long len;
	atomic_t refcount;
	struct hlist_node hnode;
	struct lksm_region *next;
	struct lksm_region *prev;
	union {
		unsigned long *filter;
		unsigned long single_filter;
	};
};

/*
 * lksm_region_ref:
 * Contains references from processes to regions
 */

struct lksm_region_ref {
	struct list_head list; /* listed by mm_slot */
	struct lksm_region *region;
};

/* the number of registered lksm_regions */
static unsigned int lksm_nr_regions;

/* the upper limit for region lookup */
#define LKSM_REGION_ITER_MAX 8

#define lksm_region_size(start, end) ((end - start) >> PAGE_SHIFT)
#define lksm_bitmap_size(size) ((size >> 6) + ((size % BITS_PER_LONG) ? 1 : 0))

/* all processes share one lksm_region for their heaps */
static struct lksm_region heap_region, unknown_region;

static void lksm_register_file_anon_region(struct mm_slot *slot,
			struct vm_area_struct *vma);
static struct lksm_region *lksm_find_region(struct vm_area_struct *vma);
#endif /* CONFIG_LKSM_FILTER */

static int initial_round = 3;
static unsigned long ksm_crawl_round;
static unsigned long crawler_sleep;

/* statistical information */
static int lksm_nr_merged; /* global merge count */
static int lksm_nr_broken; /* global broken count */
static int lksm_nr_scanned_slot; /* global scanned slot count */
static int lksm_slot_nr_merged; /* per-slot merge count */
static int lksm_slot_nr_broken; /* per-slot broken count */

/* initially, KSM takes small full scan interval */
#define DEFAULT_FULL_SCAN_INTERVAL 60000 /* 60 seconds */
static unsigned int full_scan_interval = 100;

/* statistical information about scanning time */
static unsigned long lksm_last_scan_time;
static unsigned long lksm_proc_scan_time;

/* stuffs for pruning short-lived task */
#define KSM_SHORT_TASK_TIME 100
static unsigned long short_lived_thresh = KSM_SHORT_TASK_TIME;

#define get_task_runtime(task) (task->se.sum_exec_runtime)
#define ms_to_ns(ms) (ms * 1000 * 1000)
#define check_short_task(task) \
	(get_task_runtime(task) < ms_to_ns(short_lived_thresh))

static void wait_while_offlining(void);
static struct mm_slot *__ksm_enter_alloc_slot(struct mm_struct *mm, int frozen);

static DECLARE_WAIT_QUEUE_HEAD(ksm_thread_wait);
static DECLARE_WAIT_QUEUE_HEAD(ksm_iter_wait);
static DEFINE_MUTEX(ksm_thread_mutex);
static DEFINE_SPINLOCK(ksm_mmlist_lock);
static DECLARE_WAIT_QUEUE_HEAD(ksm_crawl_wait);

#define KSM_KMEM_CACHE(__struct, __flags) kmem_cache_create("ksm_"#__struct,\
		sizeof(struct __struct), __alignof__(struct __struct),\
		(__flags), NULL)

static int __init ksm_slab_init(void)
{
	rmap_item_cache = KSM_KMEM_CACHE(rmap_item, 0);
	if (!rmap_item_cache)
		goto out;

	stable_node_cache = KSM_KMEM_CACHE(stable_node, 0);
	if (!stable_node_cache)
		goto out_free1;

	mm_slot_cache = KSM_KMEM_CACHE(mm_slot, 0);
	if (!mm_slot_cache)
		goto out_free2;
	task_slot_cache = KSM_KMEM_CACHE(task_slot, 0);
	if (!task_slot_cache)
		goto out_free3;

	return 0;

out_free3:
	kmem_cache_destroy(mm_slot_cache);
out_free2:
	kmem_cache_destroy(stable_node_cache);
out_free1:
	kmem_cache_destroy(rmap_item_cache);
out:
	return -ENOMEM;
}

static void __init ksm_slab_free(void)
{
	kmem_cache_destroy(mm_slot_cache);
	kmem_cache_destroy(stable_node_cache);
	kmem_cache_destroy(rmap_item_cache);
	mm_slot_cache = NULL;
}

static __always_inline bool is_stable_node_chain(struct stable_node *chain)
{
	return chain->rmap_hlist_len == STABLE_NODE_CHAIN;
}

static __always_inline bool is_stable_node_dup(struct stable_node *dup)
{
	return dup->head == STABLE_NODE_DUP_HEAD;
}

static inline void stable_node_chain_add_dup(struct stable_node *dup,
					     struct stable_node *chain)
{
	VM_BUG_ON(is_stable_node_dup(dup));
	dup->head = STABLE_NODE_DUP_HEAD;
	VM_BUG_ON(!is_stable_node_chain(chain));
	hlist_add_head(&dup->hlist_dup, &chain->hlist);
	ksm_stable_node_dups++;
}

static inline void __stable_node_dup_del(struct stable_node *dup)
{
	VM_BUG_ON(!is_stable_node_dup(dup));
	hlist_del(&dup->hlist_dup);
	ksm_stable_node_dups--;
}

static inline void stable_node_dup_del(struct stable_node *dup)
{
	VM_BUG_ON(is_stable_node_chain(dup));
	if (is_stable_node_dup(dup))
		__stable_node_dup_del(dup);
	else
		rb_erase(&dup->node, root_stable_tree + NUMA(dup->nid));
#ifdef CONFIG_DEBUG_VM
	dup->head = NULL;
#endif
}

static inline struct rmap_item *alloc_rmap_item(void)
{
	struct rmap_item *rmap_item;

	rmap_item = kmem_cache_zalloc(rmap_item_cache, GFP_KERNEL |
						__GFP_NORETRY | __GFP_NOWARN);
	if (rmap_item)
		ksm_rmap_items++;
	return rmap_item;
}

static inline void free_rmap_item(struct rmap_item *rmap_item)
{
	ksm_rmap_items--;
	rmap_item->mm = NULL;	/* debug safety */
	kmem_cache_free(rmap_item_cache, rmap_item);
}

static inline struct stable_node *alloc_stable_node(void)
{
	/*
	 * The allocation can take too long with GFP_KERNEL when memory is under
	 * pressure, which may lead to hung task warnings.  Adding __GFP_HIGH
	 * grants access to memory reserves, helping to avoid this problem.
	 */
	return kmem_cache_alloc(stable_node_cache, GFP_KERNEL | __GFP_HIGH);
}

static inline void free_stable_node(struct stable_node *stable_node)
{
	VM_BUG_ON(stable_node->rmap_hlist_len &&
		  !is_stable_node_chain(stable_node));
	kmem_cache_free(stable_node_cache, stable_node);
}

static inline struct mm_slot *alloc_mm_slot(void)
{
	if (!mm_slot_cache)	/* initialization failed */
		return NULL;
	return kmem_cache_zalloc(mm_slot_cache, GFP_KERNEL);
}

static inline void free_mm_slot(struct mm_slot *mm_slot)
{
	kmem_cache_free(mm_slot_cache, mm_slot);
}

static struct mm_slot *get_mm_slot(struct mm_struct *mm)
{
	struct mm_slot *slot;

	hash_for_each_possible(mm_slots_hash, slot, link, (unsigned long)mm)
		if (slot->mm == mm)
			return slot;

	return NULL;
}

static void insert_to_mm_slots_hash(struct mm_struct *mm,
				    struct mm_slot *mm_slot)
{
	mm_slot->mm = mm;
	hash_add(mm_slots_hash, &mm_slot->link, (unsigned long)mm);
}

static inline struct task_slot *alloc_task_slot(void)
{
	if (!task_slot_cache)
		return NULL;
	return kmem_cache_zalloc(task_slot_cache, GFP_NOWAIT);
}

static inline void free_task_slot(struct task_slot *task_slot)
{
	kmem_cache_free(task_slot_cache, task_slot);
}

static struct task_slot *get_task_slot(struct task_struct *task)
{
	struct task_slot *slot;

	hash_for_each_possible(task_slots_hash, slot, hlist,
			(unsigned long)task)
		if (slot->task == task)
			return slot;
	return NULL;
}

static inline void insert_to_task_slots_hash(struct task_slot *slot)
{
	hash_add(task_slots_hash, &slot->hlist, (unsigned long)slot->task);
}

/*
 * ksmd, and unmerge_and_remove_all_rmap_items(), must not touch an mm's
 * page tables after it has passed through ksm_exit() - which, if necessary,
 * takes mmap_lock briefly to serialize against them.  ksm_exit() does not set
 * a special flag: they can just back out as soon as mm_users goes to zero.
 * ksm_test_exit() is used throughout to make this test for exit: in some
 * places for correctness, in some places just to avoid unnecessary work.
 */
static inline bool ksm_test_exit(struct mm_struct *mm)
{
	return atomic_read(&mm->mm_users) == 0;
}

/*
 * We use break_ksm to break COW on a ksm page: it's a stripped down
 *
 *	if (get_user_pages(addr, 1, 1, 1, &page, NULL) == 1)
 *		put_page(page);
 *
 * but taking great care only to touch a ksm page, in a VM_MERGEABLE vma,
 * in case the application has unmapped and remapped mm,addr meanwhile.
 * Could a ksm page appear anywhere else?  Actually yes, in a VM_PFNMAP
 * mmap of /dev/mem or /dev/kmem, where we would not want to touch it.
 *
 * FAULT_FLAG/FOLL_REMOTE are because we do this outside the context
 * of the process that owns 'vma'.  We also do not want to enforce
 * protection keys here anyway.
 */
static int break_ksm(struct vm_area_struct *vma, unsigned long addr)
{
	struct page *page;
	vm_fault_t ret = 0;

	do {
		cond_resched();
		page = follow_page(vma, addr,
				FOLL_GET | FOLL_MIGRATION | FOLL_REMOTE);
		if (IS_ERR_OR_NULL(page))
			break;
		if (PageKsm(page))
			ret = handle_mm_fault(vma, addr,
					FAULT_FLAG_WRITE | FAULT_FLAG_REMOTE, NULL);
		else
			ret = VM_FAULT_WRITE;
		put_page(page);
	} while (!(ret & (VM_FAULT_WRITE | VM_FAULT_SIGBUS | VM_FAULT_SIGSEGV | VM_FAULT_OOM)));
	/*
	 * We must loop because handle_mm_fault() may back out if there's
	 * any difficulty e.g. if pte accessed bit gets updated concurrently.
	 *
	 * VM_FAULT_WRITE is what we have been hoping for: it indicates that
	 * COW has been broken, even if the vma does not permit VM_WRITE;
	 * but note that a concurrent fault might break PageKsm for us.
	 *
	 * VM_FAULT_SIGBUS could occur if we race with truncation of the
	 * backing file, which also invalidates anonymous pages: that's
	 * okay, that truncation will have unmapped the PageKsm for us.
	 *
	 * VM_FAULT_OOM: at the time of writing (late July 2009), setting
	 * aside mem_cgroup limits, VM_FAULT_OOM would only be set if the
	 * current task has TIF_MEMDIE set, and will be OOM killed on return
	 * to user; and ksmd, having no mm, would never be chosen for that.
	 *
	 * But if the mm is in a limited mem_cgroup, then the fault may fail
	 * with VM_FAULT_OOM even if the current task is not TIF_MEMDIE; and
	 * even ksmd can fail in this way - though it's usually breaking ksm
	 * just to undo a merge it made a moment before, so unlikely to oom.
	 *
	 * That's a pity: we might therefore have more kernel pages allocated
	 * than we're counting as nodes in the stable tree; but ksm_do_scan
	 * will retry to break_cow on each pass, so should recover the page
	 * in due course.  The important thing is to not let VM_MERGEABLE
	 * be cleared while any such pages might remain in the area.
	 */
	return (ret & VM_FAULT_OOM) ? -ENOMEM : 0;
}

static struct vm_area_struct *find_mergeable_vma(struct mm_struct *mm,
		unsigned long addr)
{
	struct vm_area_struct *vma;
	if (ksm_test_exit(mm))
		return NULL;
	vma = vma_lookup(mm, addr);
	if (!vma || !(vma->vm_flags & VM_MERGEABLE) || !vma->anon_vma)
		return NULL;
	return vma;
}

static void break_cow(struct rmap_item *rmap_item)
{
	struct mm_struct *mm = rmap_item->mm;
	unsigned long addr = rmap_item->address;
	struct vm_area_struct *vma;

	/*
	 * It is not an accident that whenever we want to break COW
	 * to undo, we also need to drop a reference to the anon_vma.
	 */
	put_anon_vma(rmap_item->anon_vma);

	down_read(&mm->mmap_lock);
	vma = find_mergeable_vma(mm, addr);
	if (vma)
		break_ksm(vma, addr);
	up_read(&mm->mmap_lock);
}

static struct page *get_mergeable_page(struct rmap_item *rmap_item)
{
	struct mm_struct *mm = rmap_item->mm;
	unsigned long addr = rmap_item->address;
	struct vm_area_struct *vma;
	struct page *page;

	down_read(&mm->mmap_lock);
	vma = find_mergeable_vma(mm, addr);
	if (!vma)
		goto out;

	page = follow_page(vma, addr, FOLL_GET);
	if (IS_ERR_OR_NULL(page))
		goto out;
	if (PageAnon(page)) {
		flush_anon_page(vma, page, addr);
		flush_dcache_page(page);
	} else {
		put_page(page);
out:
		page = NULL;
	}
	up_read(&mm->mmap_lock);
	return page;
}

#ifdef CONFIG_LKSM_FILTER
static inline int is_heap(struct vm_area_struct *vma)
{
	return vma->vm_start <= vma->vm_mm->brk &&
		vma->vm_end >= vma->vm_mm->start_brk;
}

/* below code is copied from fs/proc/task_mmu.c */

static int is_stack(struct vm_area_struct *vma)
{
	return vma->vm_start <= vma->vm_mm->start_stack &&
		vma->vm_end >= vma->vm_mm->start_stack;
}

static int is_exec(struct vm_area_struct *vma)
{
	return (vma->vm_flags & VM_EXEC);
}
#endif /* CONFIG_LKSM_FILTER */

/*
 * ksm_join: a wrapper function of ksm_enter.
 * The function sets VM_MERGEABLE flag of vmas in the given mm_struct.
 */
static int ksm_join(struct mm_struct *mm, int frozen)
{
	struct vm_area_struct *vma;
	struct mm_slot *slot;
	int newly_allocated = 0;

	if (!test_bit(MMF_VM_MERGEABLE, &mm->flags)) {
		slot = __ksm_enter_alloc_slot(mm, frozen);
		if (!slot)
			return -ENOMEM;
		newly_allocated = 1;
	} else {
		slot = get_mm_slot(mm);
		if (!slot) {
			ksm_err("there is no mm_slot for %p", mm);
			return -EINVAL;
		}
	}

	for (vma = mm->mmap; vma; vma = vma->vm_next) {
		if (vma->vm_flags & (VM_MERGEABLE | VM_SHARED | VM_MAYSHARE |
				VM_PFNMAP | VM_IO | VM_DONTEXPAND |
				VM_HUGETLB | VM_MIXEDMAP))
			continue;
		vma->vm_flags |= VM_MERGEABLE;
#ifdef CONFIG_LKSM_FILTER
		/*
		 * Many page sharings come from library pages because processes
		 * are sharing runtime framwork of the OS.
		 * Thus, anonymous pages related with file-mapped areas can show
		 * sharing patterns which can be exploited in LKSM while other
		 * anonymous regions (e.g., heap) don't.
		 * LKSM only tracks file-related regions to make filters.
		 */
		if (!is_heap(vma) && !is_stack(vma) &&
				!is_exec(vma) && vma->anon_vma)
			lksm_register_file_anon_region(slot, vma);
#endif
	}

	return newly_allocated;
}

#define ksm_join_write_lock(mm, frozen, ret) do {\
	down_write(&mm->mmap_lock);	\
	ret = ksm_join(mm, frozen);	\
	up_write(&mm->mmap_lock);	\
} while (0)

#ifdef CONFIG_LKSM_FILTER
static void lksm_region_ref_append
(struct mm_slot *slot, struct lksm_region *region)
{
	struct lksm_region_ref *ref;

	BUG_ON(!region);
	ref = kzalloc(sizeof(struct lksm_region_ref), GFP_KERNEL);
	if (!ref)
		return;
	ref->region = region;
	list_add_tail(&ref->list, &slot->ref_list);

	atomic_inc(&region->refcount);
}

static void lksm_region_free(struct lksm_region *region)
{
	unsigned long flags;

	spin_lock_irqsave(&lksm_region_lock, flags);
	if (!region->next) {
		if (region->prev) {
			if (atomic_read(&region->prev->refcount) == 0) {
				hash_del(&region->prev->hnode);
				if (region->prev->len > SINGLE_FILTER_LEN)
					kfree(region->prev->filter);
				kfree(region->prev);
			} else
				region->prev->next = NULL;
		}
		hash_del(&region->hnode);
		if (region->len > SINGLE_FILTER_LEN)
			kfree(region->filter);
		kfree(region);
	}
	spin_unlock_irqrestore(&lksm_region_lock, flags);
}

static void lksm_region_ref_remove(struct lksm_region_ref *ref)
{
	list_del_init(&ref->list);
	if (atomic_dec_and_test(&ref->region->refcount))
		lksm_region_free(ref->region);
	kfree(ref);
}

static void lksm_region_ref_list_release(struct mm_slot *slot)
{
	struct lksm_region_ref *ref, *next;

	list_for_each_entry_safe(ref, next, &slot->ref_list, list) {
		lksm_region_ref_remove(ref);
	}
}
#endif /* CONFIG_LKSM_FILTER */

/*
 * This helper is used for getting right index into array of tree roots.
 * When merge_across_nodes knob is set to 1, there are only two rb-trees for
 * stable and unstable pages from all nodes with roots in index 0. Otherwise,
 * every node has its own stable and unstable tree.
 */
static inline int get_kpfn_nid(unsigned long kpfn)
{
	return ksm_merge_across_nodes ? 0 : NUMA(pfn_to_nid(kpfn));
}

static struct stable_node *alloc_stable_node_chain(struct stable_node *dup,
						   struct rb_root *root)
{
	struct stable_node *chain = alloc_stable_node();
	VM_BUG_ON(is_stable_node_chain(dup));
	if (likely(chain)) {
		INIT_HLIST_HEAD(&chain->hlist);
		chain->chain_prune_time = jiffies;
		chain->rmap_hlist_len = STABLE_NODE_CHAIN;
#if defined (CONFIG_DEBUG_VM) && defined(CONFIG_NUMA)
		chain->nid = NUMA_NO_NODE; /* debug */
#endif
		ksm_stable_node_chains++;

		/*
		 * Put the stable node chain in the first dimension of
		 * the stable tree and at the same time remove the old
		 * stable node.
		 */
		rb_replace_node(&dup->node, &chain->node, root);

		/*
		 * Move the old stable node to the second dimension
		 * queued in the hlist_dup. The invariant is that all
		 * dup stable_nodes in the chain->hlist point to pages
		 * that are wrprotected and have the exact same
		 * content.
		 */
		stable_node_chain_add_dup(dup, chain);
	}
	return chain;
}

static inline void free_stable_node_chain(struct stable_node *chain,
					  struct rb_root *root)
{
	rb_erase(&chain->node, root);
	free_stable_node(chain);
	ksm_stable_node_chains--;
}

static void remove_node_from_stable_tree(struct stable_node *stable_node)
{
	struct rmap_item *rmap_item;

	/* check it's not STABLE_NODE_CHAIN or negative */
	BUG_ON(stable_node->rmap_hlist_len < 0);

	hlist_for_each_entry(rmap_item, &stable_node->hlist, hlist) {
		if (rmap_item->hlist.next) {
			ksm_pages_sharing--;
			lksm_slot_nr_broken++;
			lksm_nr_broken++;
		} else
			ksm_pages_shared--;
		VM_BUG_ON(stable_node->rmap_hlist_len <= 0);
		stable_node->rmap_hlist_len--;
		put_anon_vma(rmap_item->anon_vma);
		rmap_item->address &= PAGE_MASK;
		cond_resched();
	}

	/*
	 * We need the second aligned pointer of the migrate_nodes
	 * list_head to stay clear from the rb_parent_color union
	 * (aligned and different than any node) and also different
	 * from &migrate_nodes. This will verify that future list.h changes
	 * don't break STABLE_NODE_DUP_HEAD. Only recent gcc can handle it.
	 */
#if defined(GCC_VERSION) && GCC_VERSION >= 40903
	BUILD_BUG_ON(STABLE_NODE_DUP_HEAD <= &migrate_nodes);
	BUILD_BUG_ON(STABLE_NODE_DUP_HEAD >= &migrate_nodes + 1);
#endif

	if (stable_node->head == &migrate_nodes)
		list_del(&stable_node->list);
	else
		stable_node_dup_del(stable_node);
	free_stable_node(stable_node);
}

enum get_ksm_page_flags {
	GET_KSM_PAGE_NOLOCK,
	GET_KSM_PAGE_LOCK,
	GET_KSM_PAGE_TRYLOCK
};

/*
 * get_ksm_page: checks if the page indicated by the stable node
 * is still its ksm page, despite having held no reference to it.
 * In which case we can trust the content of the page, and it
 * returns the gotten page; but if the page has now been zapped,
 * remove the stale node from the stable tree and return NULL.
 * But beware, the stable node's page might be being migrated.
 *
 * You would expect the stable_node to hold a reference to the ksm page.
 * But if it increments the page's count, swapping out has to wait for
 * ksmd to come around again before it can free the page, which may take
 * seconds or even minutes: much too unresponsive.  So instead we use a
 * "keyhole reference": access to the ksm page from the stable node peeps
 * out through its keyhole to see if that page still holds the right key,
 * pointing back to this stable node.  This relies on freeing a PageAnon
 * page to reset its page->mapping to NULL, and relies on no other use of
 * a page to put something that might look like our key in page->mapping.
 * is on its way to being freed; but it is an anomaly to bear in mind.
 */
static struct page *get_ksm_page(struct stable_node *stable_node,
				 enum get_ksm_page_flags flags)
{
	struct page *page;
	void *expected_mapping;
	unsigned long kpfn;

	expected_mapping = (void *)((unsigned long)stable_node |
					PAGE_MAPPING_KSM);
again:
	kpfn = READ_ONCE(stable_node->kpfn); /* Address dependency. */
	page = pfn_to_page(kpfn);
	if (READ_ONCE(page->mapping) != expected_mapping)
		goto stale;

	/*
	 * We cannot do anything with the page while its refcount is 0.
	 * Usually 0 means free, or tail of a higher-order page: in which
	 * case this node is no longer referenced, and should be freed;
	 * however, it might mean that the page is under page_ref_freeze().
	 * The __remove_mapping() case is easy, again the node is now stale;
	 * the same is in reuse_ksm_page() case; but if page is swapcache
	 * in migrate_page_move_mapping(), it might still be our page,
	 * in which case it's essential to keep the node.
	 */
	while (!get_page_unless_zero(page)) {
		/*
		 * Another check for page->mapping != expected_mapping would
		 * work here too.  We have chosen the !PageSwapCache test to
		 * optimize the common case, when the page is or is about to
		 * be freed: PageSwapCache is cleared (under spin_lock_irq)
		 * in the ref_freeze section of __remove_mapping(); but Anon
		 * page->mapping reset to NULL later, in free_pages_prepare().
		 */
		if (!PageSwapCache(page))
			goto stale;
		cpu_relax();
	}

	if (READ_ONCE(page->mapping) != expected_mapping) {
		put_page(page);
		goto stale;
	}

	if (flags == GET_KSM_PAGE_TRYLOCK) {
		if (!trylock_page(page)) {
			put_page(page);
			return ERR_PTR(-EBUSY);
		}
	} else if (flags == GET_KSM_PAGE_LOCK)
		lock_page(page);

	if (flags != GET_KSM_PAGE_NOLOCK) {
		if (READ_ONCE(page->mapping) != expected_mapping) {
			unlock_page(page);
			put_page(page);
			goto stale;
		}
	}
	return page;

stale:
	/*
	 * We come here from above when page->mapping or !PageSwapCache
	 * suggests that the node is stale; but it might be under migration.
	 * We need smp_rmb(), matching the smp_wmb() in ksm_migrate_page(),
	 * before checking whether node->kpfn has been changed.
	 */
	smp_rmb();
	if (READ_ONCE(stable_node->kpfn) != kpfn)
		goto again;
	remove_node_from_stable_tree(stable_node);
	return NULL;
}

/*
 * Removing rmap_item from stable or unstable tree.
 * This function will clean the information from the stable/unstable tree.
 */
static void remove_rmap_item_from_tree(struct rmap_item *rmap_item)
{
	if (rmap_item->address & STABLE_FLAG) {
		struct stable_node *stable_node;
		struct page *page;

		stable_node = rmap_item->head;
		page = get_ksm_page(stable_node, GET_KSM_PAGE_LOCK);
		if (!page)
			goto out;

		hlist_del(&rmap_item->hlist);
		unlock_page(page);
		put_page(page);

		if (!hlist_empty(&stable_node->hlist)) {
			ksm_pages_sharing--;
			lksm_slot_nr_broken++;
			lksm_nr_broken++;
		} else
			ksm_pages_shared--;
		VM_BUG_ON(stable_node->rmap_hlist_len <= 0);
		stable_node->rmap_hlist_len--;

		put_anon_vma(rmap_item->anon_vma);
		rmap_item->head = NULL;
		rmap_item->address &= PAGE_MASK;

	} else if (rmap_item->address & UNSTABLE_FLAG) {
		unsigned char age;
		/*
		 * Usually ksmd can and must skip the rb_erase, because
		 * root_unstable_tree was already reset to RB_ROOT.
		 * But be careful when an mm is exiting: do the rb_erase
		 * if this rmap_item was inserted by this scan, rather
		 * than left over from before.
		 */
		age = (unsigned char)(ksm_scan.scan_round - rmap_item->address);
		if (!age)
			rb_erase(&rmap_item->node,
				 root_unstable_tree + NUMA(rmap_item->nid));
		else
			RB_CLEAR_NODE(&rmap_item->node);

		ksm_pages_unshared--;
		rmap_item->address &= PAGE_MASK;
	}
out:
	cond_resched();		/* we're called from many long loops */
}

static void remove_trailing_rmap_items(struct rmap_item **rmap_list)
{
	while (*rmap_list) {
		struct rmap_item *rmap_item = *rmap_list;
		*rmap_list = rmap_item->rmap_list;
		remove_rmap_item_from_tree(rmap_item);
		free_rmap_item(rmap_item);
	}
}

/*
 * Though it's very tempting to unmerge rmap_items from stable tree rather
 * than check every pte of a given vma, the locking doesn't quite work for
 * that - an rmap_item is assigned to the stable tree after inserting ksm
 * page and upping mmap_lock.  Nor does it fit with the way we skip dup'ing
 * rmap_items from parent to child at fork time (so as not to waste time
 * if exit comes before the next scan reaches it).
 *
 * Similarly, although we'd like to remove rmap_items (so updating counts
 * and freeing memory) when unmerging an area, it's easier to leave that
 * to the next pass of ksmd - consider, for example, how ksmd might be
 * in cmp_and_merge_page on one of the rmap_items we would be removing.
 */
static int unmerge_ksm_pages(struct vm_area_struct *vma,
			     unsigned long start, unsigned long end)
{
	unsigned long addr;
	int err = 0;

	for (addr = start; addr < end && !err; addr += PAGE_SIZE) {
		if (ksm_test_exit(vma->vm_mm))
			break;
		if (signal_pending(current))
			err = -ERESTARTSYS;
		else
			err = break_ksm(vma, addr);
	}
	return err;
}

static inline struct stable_node *page_stable_node(struct page *page)
{
	return PageKsm(page) ? page_rmapping(page) : NULL;
}

static inline void set_page_stable_node(struct page *page,
					struct stable_node *stable_node)
{
	page->mapping = (void *)((unsigned long)stable_node | PAGE_MAPPING_KSM);
}

#ifdef CONFIG_SYSFS
/*
 * Only called through the sysfs control interface:
 */
static int remove_stable_node(struct stable_node *stable_node)
{
	struct page *page;
	int err;

	page = get_ksm_page(stable_node, GET_KSM_PAGE_LOCK);
	if (!page) {
		/*
		 * get_ksm_page did remove_node_from_stable_tree itself.
		 */
		return 0;
	}

	/*
	 * Page could be still mapped if this races with __mmput() running in
	 * between ksm_exit() and exit_mmap(). Just refuse to let
	 * merge_across_nodes/max_page_sharing be switched.
	 */
	err = -EBUSY;
	if (!page_mapped(page)) {
		/*
		 * The stable node did not yet appear stale to get_ksm_page(),
		 * since that allows for an unmapped ksm page to be recognized
		 * right up until it is freed; but the node is safe to remove.
		 * This page might be in a pagevec waiting to be freed,
		 * or it might be PageSwapCache (perhaps under writeback),
		 * or it might have been removed from swapcache a moment ago.
		 */
		set_page_stable_node(page, NULL);
		remove_node_from_stable_tree(stable_node);
		err = 0;
	}

	unlock_page(page);
	put_page(page);
	return err;
}

static int remove_stable_node_chain(struct stable_node *stable_node,
				    struct rb_root *root)
{
	struct stable_node *dup;
	struct hlist_node *hlist_safe;

	if (!is_stable_node_chain(stable_node)) {
		VM_BUG_ON(is_stable_node_dup(stable_node));
		if (remove_stable_node(stable_node))
			return true;
		else
			return false;
	}

	hlist_for_each_entry_safe(dup, hlist_safe,
				  &stable_node->hlist, hlist_dup) {
		VM_BUG_ON(!is_stable_node_dup(dup));
		if (remove_stable_node(dup))
			return true;
	}
	BUG_ON(!hlist_empty(&stable_node->hlist));
	free_stable_node_chain(stable_node, root);
	return false;
}

static int remove_all_stable_nodes(void)
{
	struct stable_node *stable_node, *next;
	int nid;
	int err = 0;

	for (nid = 0; nid < ksm_nr_node_ids; nid++) {
		while (root_stable_tree[nid].rb_node) {
			stable_node = rb_entry(root_stable_tree[nid].rb_node,
						struct stable_node, node);
			if (remove_stable_node_chain(stable_node,
						     root_stable_tree + nid)) {
				err = -EBUSY;
				break;	/* proceed to next nid */
			}
			cond_resched();
		}
	}
	list_for_each_entry_safe(stable_node, next, &migrate_nodes, list) {
		if (remove_stable_node(stable_node))
			err = -EBUSY;
		cond_resched();
	}
	return err;
}

static int unmerge_and_remove_all_rmap_items(void)
{
	struct mm_slot *mm_slot;
	struct mm_struct *mm;
	struct vm_area_struct *vma;
	int err = 0;

	spin_lock(&ksm_mmlist_lock);
	ksm_scan.mm_slot = list_entry(ksm_mm_head.mm_list.next,
						struct mm_slot, mm_list);
	spin_unlock(&ksm_mmlist_lock);

	for (mm_slot = ksm_scan.mm_slot;
			mm_slot != &ksm_mm_head; mm_slot = ksm_scan.mm_slot) {
		mm = mm_slot->mm;
		down_read(&mm->mmap_lock);
		for (vma = mm->mmap; vma; vma = vma->vm_next) {
			if (ksm_test_exit(mm))
				break;
			if (!(vma->vm_flags & VM_MERGEABLE) || !vma->anon_vma)
				continue;
			err = unmerge_ksm_pages(vma,
						vma->vm_start, vma->vm_end);
			if (err)
				goto error;
		}

		remove_trailing_rmap_items(&mm_slot->rmap_list);
		up_read(&mm->mmap_lock);

		spin_lock(&ksm_mmlist_lock);
		ksm_scan.mm_slot = list_entry(mm_slot->mm_list.next,
						struct mm_slot, mm_list);
		if (ksm_test_exit(mm)) {
			hash_del(&mm_slot->link);
			list_del(&mm_slot->mm_list);
			spin_unlock(&ksm_mmlist_lock);

			free_mm_slot(mm_slot);
			clear_bit(MMF_VM_MERGEABLE, &mm->flags);
			mmdrop(mm);
		} else
			spin_unlock(&ksm_mmlist_lock);
	}

	/* Clean up stable nodes, but don't worry if some are still busy */
	remove_all_stable_nodes();
	ksm_scan.scan_round = 0;
	return 0;

error:
	up_read(&mm->mmap_lock);
	spin_lock(&ksm_mmlist_lock);
	ksm_scan.mm_slot = &ksm_mm_head;
	spin_unlock(&ksm_mmlist_lock);
	return err;
}
#endif /* CONFIG_SYSFS */

static u32 calc_checksum(struct page *page)
{
	u32 checksum;
	void *addr = kmap_atomic(page);
	checksum = xxhash(addr, PAGE_SIZE, 0);
	kunmap_atomic(addr);
	return lksm_clear_checksum_frozen(checksum);
}

static int write_protect_page(struct vm_area_struct *vma, struct page *page,
			      pte_t *orig_pte)
{
	struct mm_struct *mm = vma->vm_mm;
	struct page_vma_mapped_walk pvmw = {
		.page = page,
		.vma = vma,
	};
	int swapped;
	int err = -EFAULT;
	struct mmu_notifier_range range;

	pvmw.address = page_address_in_vma(page, vma);
	if (pvmw.address == -EFAULT)
		goto out;

	BUG_ON(PageTransCompound(page));

	mmu_notifier_range_init(&range, MMU_NOTIFY_CLEAR, 0, vma, mm,
				pvmw.address,
				pvmw.address + PAGE_SIZE);
	mmu_notifier_invalidate_range_start(&range);

	if (!page_vma_mapped_walk(&pvmw))
		goto out_mn;
	if (WARN_ONCE(!pvmw.pte, "Unexpected PMD mapping?"))
		goto out_unlock;

	if (pte_write(*pvmw.pte) || pte_dirty(*pvmw.pte) ||
	    (pte_protnone(*pvmw.pte) && pte_savedwrite(*pvmw.pte)) ||
						mm_tlb_flush_pending(mm)) {
		pte_t entry;

		swapped = PageSwapCache(page);
		flush_cache_page(vma, pvmw.address, page_to_pfn(page));
		/*
		 * Ok this is tricky, when get_user_pages_fast() run it doesn't
		 * take any lock, therefore the check that we are going to make
		 * with the pagecount against the mapcount is racey and
		 * O_DIRECT can happen right after the check.
		 * So we clear the pte and flush the tlb before the check
		 * this assure us that no O_DIRECT can happen after the check
		 * or in the middle of the check.
		 *
		 * No need to notify as we are downgrading page table to read
		 * only not changing it to point to a new page.
		 *
		 * See Documentation/vm/mmu_notifier.rst
		 */
		entry = ptep_clear_flush(vma, pvmw.address, pvmw.pte);
		/*
		 * Check that no O_DIRECT or similar I/O is in progress on the
		 * page
		 */
		if (page_mapcount(page) + 1 + swapped != page_count(page)) {
			set_pte_at(mm, pvmw.address, pvmw.pte, entry);
			goto out_unlock;
		}
		if (pte_dirty(entry))
			set_page_dirty(page);

		if (pte_protnone(entry))
			entry = pte_mkclean(pte_clear_savedwrite(entry));
		else
			entry = pte_mkclean(pte_wrprotect(entry));
		set_pte_at_notify(mm, pvmw.address, pvmw.pte, entry);
	}
	*orig_pte = *pvmw.pte;
	err = 0;

out_unlock:
	page_vma_mapped_walk_done(&pvmw);
out_mn:
	mmu_notifier_invalidate_range_end(&range);
out:
	return err;
}

/**
 * replace_page - replace page in vma by new ksm page
 * @vma:      vma that holds the pte pointing to page
 * @page:     the page we are replacing by kpage
 * @kpage:    the ksm page we replace page by
 * @orig_pte: the original value of the pte
 *
 * Returns 0 on success, -EFAULT on failure.
 */
static int replace_page(struct vm_area_struct *vma, struct page *page,
			struct page *kpage, pte_t orig_pte)
{
	struct mm_struct *mm = vma->vm_mm;
	pmd_t *pmd;
	pte_t *ptep;
	pte_t newpte;
	spinlock_t *ptl;
	unsigned long addr;
	int err = -EFAULT;
	struct mmu_notifier_range range;

	addr = page_address_in_vma(page, vma);
	if (addr == -EFAULT)
		goto out;

	pmd = mm_find_pmd(mm, addr);
	if (!pmd)
		goto out;

	mmu_notifier_range_init(&range, MMU_NOTIFY_CLEAR, 0, vma, mm, addr,
				addr + PAGE_SIZE);
	mmu_notifier_invalidate_range_start(&range);

	ptep = pte_offset_map_lock(mm, pmd, addr, &ptl);
	if (!pte_same(*ptep, orig_pte)) {
		pte_unmap_unlock(ptep, ptl);
		goto out_mn;
	}

	/*
	 * No need to check ksm_use_zero_pages here: we can only have a
	 * zero_page here if ksm_use_zero_pages was enabled alreaady.
	 */
	if (!is_zero_pfn(page_to_pfn(kpage))) {
		get_page(kpage);
		page_add_anon_rmap(kpage, vma, addr, false);
		newpte = mk_pte(kpage, vma->vm_page_prot);
	} else {
		newpte = pte_mkspecial(pfn_pte(page_to_pfn(kpage),
					       vma->vm_page_prot));
		/*
		 * We're replacing an anonymous page with a zero page, which is
		 * not anonymous. We need to do proper accounting otherwise we
		 * will get wrong values in /proc, and a BUG message in dmesg
		 * when tearing down the mm.
		 */
		dec_mm_counter(mm, MM_ANONPAGES);
	}

	flush_cache_page(vma, addr, pte_pfn(*ptep));
	/*
	 * No need to notify as we are replacing a read only page with another
	 * read only page with the same content.
	 *
	 * See Documentation/vm/mmu_notifier.rst
	 */
	ptep_clear_flush(vma, addr, ptep);
	set_pte_at_notify(mm, addr, ptep, newpte);

	page_remove_rmap(page, false);
	if (!page_mapped(page))
		try_to_free_swap(page);
	put_page(page);

	pte_unmap_unlock(ptep, ptl);
	err = 0;
out_mn:
	mmu_notifier_invalidate_range_end(&range);
out:
	return err;
}

/*
 * try_to_merge_one_page - take two pages and merge them into one
 * @vma: the vma that holds the pte pointing to page
 * @page: the PageAnon page that we want to replace with kpage
 * @kpage: the PageKsm page that we want to map instead of page,
 *         or NULL the first time when we want to use page as kpage.
 *
 * This function returns 0 if the pages were merged, -EFAULT otherwise.
 */
static int try_to_merge_one_page(struct vm_area_struct *vma,
				 struct page *page, struct page *kpage)
{
	pte_t orig_pte = __pte(0);
	int err = -EFAULT;

	if (page == kpage)			/* ksm page forked */
		return 0;

	if (!PageAnon(page))
		goto out;

	/*
	 * We need the page lock to read a stable PageSwapCache in
	 * write_protect_page().  We use trylock_page() instead of
	 * lock_page() because we don't want to wait here - we
	 * prefer to continue scanning and merging different pages,
	 * then come back to this page when it is unlocked.
	 */
	if (!trylock_page(page))
		goto out;

	if (PageTransCompound(page)) {
		if (split_huge_page(page))
			goto out_unlock;
	}

	/*
	 * If this anonymous page is mapped only here, its pte may need
	 * to be write-protected.  If it's mapped elsewhere, all of its
	 * ptes are necessarily already write-protected.  But in either
	 * case, we need to lock and check page_count is not raised.
	 */
	if (write_protect_page(vma, page, &orig_pte) == 0) {
		if (!kpage) {
			/*
			 * While we hold page lock, upgrade page from
			 * PageAnon+anon_vma to PageKsm+NULL stable_node:
			 * stable_tree_insert() will update stable_node.
			 */
			set_page_stable_node(page, NULL);
			mark_page_accessed(page);
			/*
			 * Page reclaim just frees a clean page with no dirty
			 * ptes: make sure that the ksm page would be swapped.
			 */
			if (!PageDirty(page))
				SetPageDirty(page);
			err = 0;
		} else if (pages_identical(page, kpage))
			err = replace_page(vma, page, kpage, orig_pte);
	}

	if ((vma->vm_flags & VM_LOCKED) && kpage && !err) {
		munlock_vma_page(page);
		if (!PageMlocked(kpage)) {
			unlock_page(page);
			lock_page(kpage);
			mlock_vma_page(kpage);
			page = kpage;		/* for final unlock */
		}
	}

out_unlock:
	unlock_page(page);
out:
	return err;
}

/*
 * try_to_merge_with_ksm_page - like try_to_merge_two_pages,
 * but no new kernel page is allocated: kpage must already be a ksm page.
 *
 * This function returns 0 if the pages were merged, -EFAULT otherwise.
 */
static int try_to_merge_with_ksm_page(struct rmap_item *rmap_item,
				      struct page *page, struct page *kpage)
{
	struct mm_struct *mm = rmap_item->mm;
	struct vm_area_struct *vma;
	int err = -EFAULT;

	down_read(&mm->mmap_lock);
	vma = find_mergeable_vma(mm, rmap_item->address);
	if (!vma)
		goto out;

	err = try_to_merge_one_page(vma, page, kpage);
	if (err)
		goto out;

	/* Unstable nid is in union with stable anon_vma: remove first */
	remove_rmap_item_from_tree(rmap_item);

#ifdef CONFIG_LKSM_FILTER
	/* node is removed from tree, base_addr can be safely used */
	rmap_item->base_addr = vma->vm_start;
#endif
	/* Must get reference to anon_vma while still holding mmap_lock */
	rmap_item->anon_vma = vma->anon_vma;
	get_anon_vma(vma->anon_vma);
out:
	up_read(&mm->mmap_lock);
	return err;
}

/*
 * try_to_merge_two_pages - take two identical pages and prepare them
 * to be merged into one page.
 *
 * This function returns the kpage if we successfully merged two identical
 * pages into one ksm page, NULL otherwise.
 *
 * Note that this function upgrades page to ksm page: if one of the pages
 * is already a ksm page, try_to_merge_with_ksm_page should be used.
 */
static struct page *try_to_merge_two_pages(struct rmap_item *rmap_item,
					   struct page *page,
					   struct rmap_item *tree_rmap_item,
					   struct page *tree_page)
{
	int err;

	err = try_to_merge_with_ksm_page(rmap_item, page, NULL);
	if (!err) {
		err = try_to_merge_with_ksm_page(tree_rmap_item,
							tree_page, page);
		/*
		 * If that fails, we have a ksm page with only one pte
		 * pointing to it: so break it.
		 */
		if (err)
			break_cow(rmap_item);
	}
	return err ? NULL : page;
}

static __always_inline
bool __is_page_sharing_candidate(struct stable_node *stable_node, int offset)
{
	VM_BUG_ON(stable_node->rmap_hlist_len < 0);
	/*
	 * Check that at least one mapping still exists, otherwise
	 * there's no much point to merge and share with this
	 * stable_node, as the underlying tree_page of the other
	 * sharer is going to be freed soon.
	 */
	return stable_node->rmap_hlist_len &&
		stable_node->rmap_hlist_len + offset < ksm_max_page_sharing;
}

static __always_inline
bool is_page_sharing_candidate(struct stable_node *stable_node)
{
	return __is_page_sharing_candidate(stable_node, 0);
}

static struct page *stable_node_dup(struct stable_node **_stable_node_dup,
				    struct stable_node **_stable_node,
				    struct rb_root *root,
				    bool prune_stale_stable_nodes)
{
	struct stable_node *dup, *found = NULL, *stable_node = *_stable_node;
	struct hlist_node *hlist_safe;
	struct page *_tree_page, *tree_page = NULL;
	int nr = 0;
	int found_rmap_hlist_len;

	if (!prune_stale_stable_nodes ||
	    time_before(jiffies, stable_node->chain_prune_time +
			msecs_to_jiffies(
				ksm_stable_node_chains_prune_millisecs)))
		prune_stale_stable_nodes = false;
	else
		stable_node->chain_prune_time = jiffies;

	hlist_for_each_entry_safe(dup, hlist_safe,
				  &stable_node->hlist, hlist_dup) {
		cond_resched();
		/*
		 * We must walk all stable_node_dup to prune the stale
		 * stable nodes during lookup.
		 *
		 * get_ksm_page can drop the nodes from the
		 * stable_node->hlist if they point to freed pages
		 * (that's why we do a _safe walk). The "dup"
		 * stable_node parameter itself will be freed from
		 * under us if it returns NULL.
		 */
		_tree_page = get_ksm_page(dup, GET_KSM_PAGE_NOLOCK);
		if (!_tree_page)
			continue;
		nr += 1;
		if (is_page_sharing_candidate(dup)) {
			if (!found ||
			    dup->rmap_hlist_len > found_rmap_hlist_len) {
				if (found)
					put_page(tree_page);
				found = dup;
				found_rmap_hlist_len = found->rmap_hlist_len;
				tree_page = _tree_page;

				/* skip put_page for found dup */
				if (!prune_stale_stable_nodes)
					break;
				continue;
			}
		}
		put_page(_tree_page);
	}

	if (found) {
		/*
		 * nr is counting all dups in the chain only if
		 * prune_stale_stable_nodes is true, otherwise we may
		 * break the loop at nr == 1 even if there are
		 * multiple entries.
		 */
		if (prune_stale_stable_nodes && nr == 1) {
			/*
			 * If there's not just one entry it would
			 * corrupt memory, better BUG_ON. In KSM
			 * context with no lock held it's not even
			 * fatal.
			 */
			BUG_ON(stable_node->hlist.first->next);

			/*
			 * There's just one entry and it is below the
			 * deduplication limit so drop the chain.
			 */
			rb_replace_node(&stable_node->node, &found->node,
					root);
			free_stable_node(stable_node);
			ksm_stable_node_chains--;
			ksm_stable_node_dups--;
			/*
			 * NOTE: the caller depends on the stable_node
			 * to be equal to stable_node_dup if the chain
			 * was collapsed.
			 */
			*_stable_node = found;
			/*
			 * Just for robustneess as stable_node is
			 * otherwise left as a stable pointer, the
			 * compiler shall optimize it away at build
			 * time.
			 */
			stable_node = NULL;
		} else if (stable_node->hlist.first != &found->hlist_dup &&
			   __is_page_sharing_candidate(found, 1)) {
			/*
			 * If the found stable_node dup can accept one
			 * more future merge (in addition to the one
			 * that is underway) and is not at the head of
			 * the chain, put it there so next search will
			 * be quicker in the !prune_stale_stable_nodes
			 * case.
			 *
			 * NOTE: it would be inaccurate to use nr > 1
			 * instead of checking the hlist.first pointer
			 * directly, because in the
			 * prune_stale_stable_nodes case "nr" isn't
			 * the position of the found dup in the chain,
			 * but the total number of dups in the chain.
			 */
			hlist_del(&found->hlist_dup);
			hlist_add_head(&found->hlist_dup,
				       &stable_node->hlist);
		}
	}

	*_stable_node_dup = found;
	return tree_page;
}

static struct stable_node *stable_node_dup_any(struct stable_node *stable_node,
					       struct rb_root *root)
{
	if (!is_stable_node_chain(stable_node))
		return stable_node;
	if (hlist_empty(&stable_node->hlist)) {
		free_stable_node_chain(stable_node, root);
		return NULL;
	}
	return hlist_entry(stable_node->hlist.first,
			   typeof(*stable_node), hlist_dup);
}

/*
 * Like for get_ksm_page, this function can free the *_stable_node and
 * *_stable_node_dup if the returned tree_page is NULL.
 *
 * It can also free and overwrite *_stable_node with the found
 * stable_node_dup if the chain is collapsed (in which case
 * *_stable_node will be equal to *_stable_node_dup like if the chain
 * never existed). It's up to the caller to verify tree_page is not
 * NULL before dereferencing *_stable_node or *_stable_node_dup.
 *
 * *_stable_node_dup is really a second output parameter of this
 * function and will be overwritten in all cases, the caller doesn't
 * need to initialize it.
 */
static struct page *__stable_node_chain(struct stable_node **_stable_node_dup,
					struct stable_node **_stable_node,
					struct rb_root *root,
					bool prune_stale_stable_nodes)
{
	struct stable_node *stable_node = *_stable_node;
	if (!is_stable_node_chain(stable_node)) {
		if (is_page_sharing_candidate(stable_node)) {
			*_stable_node_dup = stable_node;
			return get_ksm_page(stable_node, GET_KSM_PAGE_NOLOCK);
		}
		/*
		 * _stable_node_dup set to NULL means the stable_node
		 * reached the ksm_max_page_sharing limit.
		 */
		*_stable_node_dup = NULL;
		return NULL;
	}
	return stable_node_dup(_stable_node_dup, _stable_node, root,
			       prune_stale_stable_nodes);
}

static __always_inline struct page *chain_prune(struct stable_node **s_n_d,
						struct stable_node **s_n,
						struct rb_root *root)
{
	return __stable_node_chain(s_n_d, s_n, root, true);
}

static __always_inline struct page *chain(struct stable_node **s_n_d,
					  struct stable_node *s_n,
					  struct rb_root *root)
{
	struct stable_node *old_stable_node = s_n;
	struct page *tree_page;

	tree_page = __stable_node_chain(s_n_d, &s_n, root, false);
	/* not pruning dups so s_n cannot have changed */
	VM_BUG_ON(s_n != old_stable_node);
	return tree_page;
}

/*
 * stable_tree_search - search for page inside the stable tree
 *
 * This function checks if there is a page inside the stable tree
 * with identical content to the page that we are scanning right now.
 *
 * This function returns the stable tree node of identical content if found,
 * NULL otherwise.
 */
static struct page *stable_tree_search(struct page *page)
{
	int nid;
	struct rb_root *root;
	struct rb_node **new;
	struct rb_node *parent;
	struct stable_node *stable_node, *stable_node_dup, *stable_node_any;
	struct stable_node *page_node;

	page_node = page_stable_node(page);
	if (page_node && page_node->head != &migrate_nodes) {
		/* ksm page forked */
		get_page(page);
		return page;
	}

	nid = get_kpfn_nid(page_to_pfn(page));
	root = root_stable_tree + nid;
again:
	new = &root->rb_node;
	parent = NULL;

	while (*new) {
		struct page *tree_page;
		int ret;

		cond_resched();
		stable_node = rb_entry(*new, struct stable_node, node);
		stable_node_any = NULL;
		tree_page = chain_prune(&stable_node_dup, &stable_node,	root);
		/*
		 * NOTE: stable_node may have been freed by
		 * chain_prune() if the returned stable_node_dup is
		 * not NULL. stable_node_dup may have been inserted in
		 * the rbtree instead as a regular stable_node (in
		 * order to collapse the stable_node chain if a single
		 * stable_node dup was found in it). In such case the
		 * stable_node is overwritten by the calleee to point
		 * to the stable_node_dup that was collapsed in the
		 * stable rbtree and stable_node will be equal to
		 * stable_node_dup like if the chain never existed.
		 */
		if (!stable_node_dup) {
			/*
			 * Either all stable_node dups were full in
			 * this stable_node chain, or this chain was
			 * empty and should be rb_erased.
			 */
			stable_node_any = stable_node_dup_any(stable_node,
							      root);
			if (!stable_node_any) {
				/* rb_erase just run */
				goto again;
			}
			/*
			 * Take any of the stable_node dups page of
			 * this stable_node chain to let the tree walk
			 * continue. All KSM pages belonging to the
			 * stable_node dups in a stable_node chain
			 * have the same content and they're
			 * wrprotected at all times. Any will work
			 * fine to continue the walk.
			 */
			tree_page = get_ksm_page(stable_node_any,
						 GET_KSM_PAGE_NOLOCK);
		}
		VM_BUG_ON(!stable_node_dup ^ !!stable_node_any);
		if (!tree_page) {
			/*
			 * If we walked over a stale stable_node,
			 * get_ksm_page() will call rb_erase() and it
			 * may rebalance the tree from under us. So
			 * restart the search from scratch. Returning
			 * NULL would be safe too, but we'd generate
			 * false negative insertions just because some
			 * stable_node was stale.
			 */
			goto again;
		}

		ret = memcmp_pages(page, tree_page);
		put_page(tree_page);

		parent = *new;
		if (ret < 0)
			new = &parent->rb_left;
		else if (ret > 0)
			new = &parent->rb_right;
		else {
			if (page_node) {
				VM_BUG_ON(page_node->head != &migrate_nodes);
				/*
				 * Test if the migrated page should be merged
				 * into a stable node dup. If the mapcount is
				 * 1 we can migrate it with another KSM page
				 * without adding it to the chain.
				 */
				if (page_mapcount(page) > 1)
					goto chain_append;
			}

			if (!stable_node_dup) {
				/*
				 * If the stable_node is a chain and
				 * we got a payload match in memcmp
				 * but we cannot merge the scanned
				 * page in any of the existing
				 * stable_node dups because they're
				 * all full, we need to wait the
				 * scanned page to find itself a match
				 * in the unstable tree to create a
				 * brand new KSM page to add later to
				 * the dups of this stable_node.
				 */
				return NULL;
			}

			/*
			 * Lock and unlock the stable_node's page (which
			 * might already have been migrated) so that page
			 * migration is sure to notice its raised count.
			 * It would be more elegant to return stable_node
			 * than kpage, but that involves more changes.
			 */
			tree_page = get_ksm_page(stable_node_dup,
						 GET_KSM_PAGE_TRYLOCK);

			if (PTR_ERR(tree_page) == -EBUSY)
				return ERR_PTR(-EBUSY);

			if (unlikely(!tree_page))
				/*
				 * The tree may have been rebalanced,
				 * so re-evaluate parent and new.
				 */
				goto again;
			unlock_page(tree_page);

			if (get_kpfn_nid(stable_node_dup->kpfn) !=
			    NUMA(stable_node_dup->nid)) {
				put_page(tree_page);
				goto replace;
			}
			return tree_page;
		}
	}

	if (!page_node)
		return NULL;

	list_del(&page_node->list);
	DO_NUMA(page_node->nid = nid);
	rb_link_node(&page_node->node, parent, new);
	rb_insert_color(&page_node->node, root);
out:
	if (is_page_sharing_candidate(page_node)) {
		get_page(page);
		return page;
	} else
		return NULL;

replace:
	/*
	 * If stable_node was a chain and chain_prune collapsed it,
	 * stable_node has been updated to be the new regular
	 * stable_node. A collapse of the chain is indistinguishable
	 * from the case there was no chain in the stable
	 * rbtree. Otherwise stable_node is the chain and
	 * stable_node_dup is the dup to replace.
	 */
	if (stable_node_dup == stable_node) {
		VM_BUG_ON(is_stable_node_chain(stable_node_dup));
		VM_BUG_ON(is_stable_node_dup(stable_node_dup));
		/* there is no chain */
		if (page_node) {
			VM_BUG_ON(page_node->head != &migrate_nodes);
			list_del(&page_node->list);
			DO_NUMA(page_node->nid = nid);
			rb_replace_node(&stable_node_dup->node,
					&page_node->node,
					root);
			if (is_page_sharing_candidate(page_node))
				get_page(page);
			else
				page = NULL;
		} else {
			rb_erase(&stable_node_dup->node, root);
			page = NULL;
		}
	} else {
		VM_BUG_ON(!is_stable_node_chain(stable_node));
		__stable_node_dup_del(stable_node_dup);
		if (page_node) {
			VM_BUG_ON(page_node->head != &migrate_nodes);
			list_del(&page_node->list);
			DO_NUMA(page_node->nid = nid);
			stable_node_chain_add_dup(page_node, stable_node);
			if (is_page_sharing_candidate(page_node))
				get_page(page);
			else
				page = NULL;
		} else {
			page = NULL;
		}
	}
	stable_node_dup->head = &migrate_nodes;
	list_add(&stable_node_dup->list, stable_node_dup->head);
	return page;

chain_append:
	/* stable_node_dup could be null if it reached the limit */
	if (!stable_node_dup)
		stable_node_dup = stable_node_any;
	/*
	 * If stable_node was a chain and chain_prune collapsed it,
	 * stable_node has been updated to be the new regular
	 * stable_node. A collapse of the chain is indistinguishable
	 * from the case there was no chain in the stable
	 * rbtree. Otherwise stable_node is the chain and
	 * stable_node_dup is the dup to replace.
	 */
	if (stable_node_dup == stable_node) {
		VM_BUG_ON(is_stable_node_dup(stable_node_dup));
		/* chain is missing so create it */
		stable_node = alloc_stable_node_chain(stable_node_dup,
						      root);
		if (!stable_node)
			return NULL;
	}
	/*
	 * Add this stable_node dup that was
	 * migrated to the stable_node chain
	 * of the current nid for this page
	 * content.
	 */
	VM_BUG_ON(!is_stable_node_dup(stable_node_dup));
	VM_BUG_ON(page_node->head != &migrate_nodes);
	list_del(&page_node->list);
	DO_NUMA(page_node->nid = nid);
	stable_node_chain_add_dup(page_node, stable_node);
	goto out;
}

/*
 * stable_tree_insert - insert stable tree node pointing to new ksm page
 * into the stable tree.
 *
 * This function returns the stable tree node just allocated on success,
 * NULL otherwise.
 */
static struct stable_node *stable_tree_insert(struct page *kpage)
{
	int nid;
	unsigned long kpfn;
	struct rb_root *root;
	struct rb_node **new;
	struct rb_node *parent;
	struct stable_node *stable_node, *stable_node_dup, *stable_node_any;
	bool need_chain = false;

	kpfn = page_to_pfn(kpage);
	nid = get_kpfn_nid(kpfn);
	root = root_stable_tree + nid;
again:
	parent = NULL;
	new = &root->rb_node;

	while (*new) {
		struct page *tree_page;
		int ret;

		cond_resched();
		stable_node = rb_entry(*new, struct stable_node, node);
		stable_node_any = NULL;
		tree_page = chain(&stable_node_dup, stable_node, root);
		if (!stable_node_dup) {
			/*
			 * Either all stable_node dups were full in
			 * this stable_node chain, or this chain was
			 * empty and should be rb_erased.
			 */
			stable_node_any = stable_node_dup_any(stable_node,
							      root);
			if (!stable_node_any) {
				/* rb_erase just run */
				goto again;
			}
			/*
			 * Take any of the stable_node dups page of
			 * this stable_node chain to let the tree walk
			 * continue. All KSM pages belonging to the
			 * stable_node dups in a stable_node chain
			 * have the same content and they're
			 * wrprotected at all times. Any will work
			 * fine to continue the walk.
			 */
			tree_page = get_ksm_page(stable_node_any,
						 GET_KSM_PAGE_NOLOCK);
		}
		VM_BUG_ON(!stable_node_dup ^ !!stable_node_any);
		if (!tree_page) {
			/*
			 * If we walked over a stale stable_node,
			 * get_ksm_page() will call rb_erase() and it
			 * may rebalance the tree from under us. So
			 * restart the search from scratch. Returning
			 * NULL would be safe too, but we'd generate
			 * false negative insertions just because some
			 * stable_node was stale.
			 */
			goto again;
		}

		ret = memcmp_pages(kpage, tree_page);
		put_page(tree_page);

		parent = *new;
		if (ret < 0)
			new = &parent->rb_left;
		else if (ret > 0)
			new = &parent->rb_right;
		else {
			need_chain = true;
			break;
		}
	}

	stable_node_dup = alloc_stable_node();
	if (!stable_node_dup)
		return NULL;

	INIT_HLIST_HEAD(&stable_node_dup->hlist);
	stable_node_dup->kpfn = kpfn;
	set_page_stable_node(kpage, stable_node_dup);
	stable_node_dup->rmap_hlist_len = 0;
	DO_NUMA(stable_node_dup->nid = nid);
	if (!need_chain) {
		rb_link_node(&stable_node_dup->node, parent, new);
		rb_insert_color(&stable_node_dup->node, root);
	} else {
		if (!is_stable_node_chain(stable_node)) {
			struct stable_node *orig = stable_node;
			/* chain is missing so create it */
			stable_node = alloc_stable_node_chain(orig, root);
			if (!stable_node) {
				free_stable_node(stable_node_dup);
				return NULL;
			}
		}
		stable_node_chain_add_dup(stable_node_dup, stable_node);
	}

	return stable_node_dup;
}

/*
 * unstable_tree_search_insert - search for identical page,
 * else insert rmap_item into the unstable tree.
 *
 * This function searches for a page in the unstable tree identical to the
 * page currently being scanned; and if no identical page is found in the
 * tree, we insert rmap_item as a new object into the unstable tree.
 *
 * This function returns pointer to rmap_item found to be identical
 * to the currently scanned page, NULL otherwise.
 *
 * This function does both searching and inserting, because they share
 * the same walking algorithm in an rbtree.
 */
static
struct rmap_item *unstable_tree_search_insert(struct rmap_item *rmap_item,
					      struct page *page,
					      struct page **tree_pagep)
{
	struct rb_node **new;
	struct rb_root *root;
	struct rb_node *parent = NULL;
	int nid;

	nid = get_kpfn_nid(page_to_pfn(page));
	root = root_unstable_tree + nid;
	new = &root->rb_node;

	while (*new) {
		struct rmap_item *tree_rmap_item;
		struct page *tree_page;
		int ret;

		cond_resched();
		tree_rmap_item = rb_entry(*new, struct rmap_item, node);
		tree_page = get_mergeable_page(tree_rmap_item);
		if (!tree_page)
			return NULL;

		/*
		 * Don't substitute a ksm page for a forked page.
		 */
		if (page == tree_page) {
			put_page(tree_page);
			return NULL;
		}

		ret = memcmp_pages(page, tree_page);

		parent = *new;
		if (ret < 0) {
			put_page(tree_page);
			new = &parent->rb_left;
		} else if (ret > 0) {
			put_page(tree_page);
			new = &parent->rb_right;
		} else if (!ksm_merge_across_nodes &&
			   page_to_nid(tree_page) != nid) {
			/*
			 * If tree_page has been migrated to another NUMA node,
			 * it will be flushed out and put in the right unstable
			 * tree next time: only merge with it when across_nodes.
			 */
			put_page(tree_page);
			return NULL;
		} else {
			*tree_pagep = tree_page;
			return tree_rmap_item;
		}
	}

	rmap_item->address |= UNSTABLE_FLAG;
	rmap_item->address |= (ksm_scan.scan_round & SEQNR_MASK);
	DO_NUMA(rmap_item->nid = nid);
	rb_link_node(&rmap_item->node, parent, new);
	rb_insert_color(&rmap_item->node, root);

#ifdef CONFIG_LKSM_FILTER
	rmap_item->region = ksm_scan.region;
#endif
	ksm_pages_unshared++;
	return NULL;
}

/*
 * stable_tree_append - add another rmap_item to the linked list of
 * rmap_items hanging off a given node of the stable tree, all sharing
 * the same ksm page.
 */
static void stable_tree_append(struct rmap_item *rmap_item,
			       struct stable_node *stable_node,
			       bool max_page_sharing_bypass)
{
	/*
	 * rmap won't find this mapping if we don't insert the
	 * rmap_item in the right stable_node
	 * duplicate. page_migration could break later if rmap breaks,
	 * so we can as well crash here. We really need to check for
	 * rmap_hlist_len == STABLE_NODE_CHAIN, but we can as well check
	 * for other negative values as an undeflow if detected here
	 * for the first time (and not when decreasing rmap_hlist_len)
	 * would be sign of memory corruption in the stable_node.
	 */
	BUG_ON(stable_node->rmap_hlist_len < 0);

	stable_node->rmap_hlist_len++;
	if (!max_page_sharing_bypass)
		/* possibly non fatal but unexpected overflow, only warn */
		WARN_ON_ONCE(stable_node->rmap_hlist_len >
			     ksm_max_page_sharing);

	rmap_item->head = stable_node;
	rmap_item->address |= STABLE_FLAG;
	hlist_add_head(&rmap_item->hlist, &stable_node->hlist);

	if (rmap_item->hlist.next) {
		ksm_pages_sharing++;
		lksm_slot_nr_merged++;
		lksm_nr_merged++;
	} else
		ksm_pages_shared++;
}

#ifdef CONFIG_LKSM_FILTER
static inline void stable_tree_append_region(struct rmap_item *rmap_item,
			       struct stable_node *stable_node,
			       struct lksm_region *region,
			       bool max_page_sharing_bypass)
{
	if (region->type == LKSM_REGION_FILE1
			|| region->type == LKSM_REGION_FILE2) {
		int ret;
		unsigned long offset =
				(rmap_item->address - rmap_item->base_addr) >> PAGE_SHIFT;
		if (unlikely(region->filter_cnt == 0)
				&& region->len > SINGLE_FILTER_LEN
				&& !region->filter) {
			region->filter = kcalloc(region->len, sizeof(long), GFP_KERNEL);
			if (!region->filter) {
				ksm_err("fail to allocate memory for filter");
				goto skip;
			}
		}
		if (region->len > SINGLE_FILTER_LEN)
			ret = test_and_set_bit(offset, region->filter);
		else
			ret = test_and_set_bit(offset, &region->single_filter);
		if (!ret)
			region->filter_cnt++;
	}
	region->merge_cnt++;
	region_share[region->type]++;
skip:
	stable_tree_append(rmap_item, stable_node, max_page_sharing_bypass);
}
#endif /* CONFIG_LKSM_FILTER */

/*
 * cmp_and_merge_page - first see if page can be merged into the stable tree;
 * if not, compare checksum to previous and if it's the same, see if page can
 * be inserted into the unstable tree, or merged with a page already there and
 * both transferred to the stable tree.
 *
 * @page: the page that we are searching identical page to.
 * @rmap_item: the reverse mapping into the virtual address of this page
 */
static void cmp_and_merge_page(struct page *page, struct rmap_item *rmap_item)
{
	struct mm_struct *mm = rmap_item->mm;
	struct rmap_item *tree_rmap_item;
	struct page *tree_page = NULL;
	struct stable_node *stable_node;
	struct page *kpage;
	unsigned int checksum;
	int err;
	bool max_page_sharing_bypass = false;

	stable_node = page_stable_node(page);
	if (stable_node) {
		if (stable_node->head != &migrate_nodes &&
		    get_kpfn_nid(READ_ONCE(stable_node->kpfn)) !=
		    NUMA(stable_node->nid)) {
			stable_node_dup_del(stable_node);
			stable_node->head = &migrate_nodes;
			list_add(&stable_node->list, stable_node->head);
		}
		if (stable_node->head != &migrate_nodes &&
		    rmap_item->head == stable_node)
			return;
		/*
		 * If it's a KSM fork, allow it to go over the sharing limit
		 * without warnings.
		 */
		if (!is_page_sharing_candidate(stable_node))
			max_page_sharing_bypass = true;
	}

	/* We first start with searching the page inside the stable tree */
	kpage = stable_tree_search(page);
	if (kpage == page && rmap_item->head == stable_node) {
		put_page(kpage);
		return;
	}

	remove_rmap_item_from_tree(rmap_item);

	if (kpage) {
		if (PTR_ERR(kpage) == -EBUSY)
			return;

		err = try_to_merge_with_ksm_page(rmap_item, page, kpage);
		if (!err) {
			/*
			 * The page was successfully merged:
			 * add its rmap_item to the stable tree.
			 */
			lock_page(kpage);
#ifdef CONFIG_LKSM_FILTER
			stable_tree_append_region(rmap_item, page_stable_node(kpage),
					ksm_scan.region, max_page_sharing_bypass);
#else
			stable_tree_append(rmap_item, page_stable_node(kpage),
					   max_page_sharing_bypass);
#endif
			unlock_page(kpage);
		}
		put_page(kpage);
		return;
	}

	/*
	 * LKSM: In LKSM, KSM is running in a event-triggered manner.
	 * Because of that scanning is much infrequently performed.
	 * We just skip caculation of checksum for LKSM to catch scanning
	 * chances more.
	 */
	if (ksm_scan.scan_round < initial_round
				&& !lksm_test_rmap_frozen(rmap_item)) {
		checksum = calc_checksum(page);
		if (rmap_item->oldchecksum != checksum) {
			rmap_item->oldchecksum = checksum;
			return;
		}
	}

	/*
	 * Same checksum as an empty page. We attempt to merge it with the
	 * appropriate zero page if the user enabled this via sysfs.
	 */
	if (ksm_use_zero_pages && (checksum == zero_checksum)) {
		struct vm_area_struct *vma;

		down_read(&mm->mmap_lock);
		vma = find_mergeable_vma(mm, rmap_item->address);
		if (vma) {
			err = try_to_merge_one_page(vma, page,
					ZERO_PAGE(rmap_item->address));
		} else {
			/*
			 * If the vma is out of date, we do not need to
			 * continue.
			 */
			err = 0;
		}
		up_read(&mm->mmap_lock);
		/*
		 * In case of failure, the page was not really empty, so we
		 * need to continue. Otherwise we're done.
		 */
		if (!err)
			return;
	}
	tree_rmap_item =
		unstable_tree_search_insert(rmap_item, page, &tree_page);
	if (tree_rmap_item) {
		bool split;
#ifdef CONFIG_LKSM_FILTER
		struct lksm_region *tree_region = tree_rmap_item->region;
#endif
		kpage = try_to_merge_two_pages(rmap_item, page,
						tree_rmap_item, tree_page);
		/*
		 * If both pages we tried to merge belong to the same compound
		 * page, then we actually ended up increasing the reference
		 * count of the same compound page twice, and split_huge_page
		 * failed.
		 * Here we set a flag if that happened, and we use it later to
		 * try split_huge_page again. Since we call put_page right
		 * afterwards, the reference count will be correct and
		 * split_huge_page should succeed.
		 */
		split = PageTransCompound(page)
			&& compound_head(page) == compound_head(tree_page);
		put_page(tree_page);
		if (kpage) {
			/*
			 * The pages were successfully merged: insert new
			 * node in the stable tree and add both rmap_items.
			 */
			lock_page(kpage);
			stable_node = stable_tree_insert(kpage);
			if (stable_node) {
#ifdef CONFIG_LKSM_FILTER
				stable_tree_append_region(tree_rmap_item, stable_node,
						tree_region, false);
				stable_tree_append_region(rmap_item, stable_node,
						ksm_scan.region, false);
#else
				stable_tree_append(tree_rmap_item, stable_node,
						   false);
				stable_tree_append(rmap_item, stable_node,
						   false);
#endif
			}
			unlock_page(kpage);

			/*
			 * If we fail to insert the page into the stable tree,
			 * we will have 2 virtual addresses that are pointing
			 * to a ksm page left outside the stable tree,
			 * in which case we need to break_cow on both.
			 */
			if (!stable_node) {
				break_cow(tree_rmap_item);
				break_cow(rmap_item);
#ifdef CONFIG_LKSM_FILTER
				tree_rmap_item->region = tree_region;
				rmap_item->region = ksm_scan.region;
#endif
			}
		} else if (split) {
			/*
			 * We are here if we tried to merge two pages and
			 * failed because they both belonged to the same
			 * compound page. We will split the page now, but no
			 * merging will take place.
			 * We do not want to add the cost of a full lock; if
			 * the page is locked, it is better to skip it and
			 * perhaps try again later.
			 */
			if (!trylock_page(page))
				return;
			split_huge_page(page);
			unlock_page(page);
		}
	}
}

static struct rmap_item *get_next_rmap_item(struct mm_slot *mm_slot,
					    struct rmap_item **rmap_list,
					    unsigned long addr)
{
	struct rmap_item *rmap_item;

	while (*rmap_list) {
		rmap_item = *rmap_list;
		if ((rmap_item->address & PAGE_MASK) == addr) {
			if (lksm_test_mm_state(mm_slot, KSM_MM_FROZEN)
					&& rmap_item->address & UNSTABLE_FLAG)
				lksm_set_rmap_frozen(rmap_item);
			else
				lksm_clear_rmap_frozen(rmap_item);
			return rmap_item;
		}
		if (rmap_item->address > addr)
			break;
		*rmap_list = rmap_item->rmap_list;
		remove_rmap_item_from_tree(rmap_item);
		free_rmap_item(rmap_item);
	}

	rmap_item = alloc_rmap_item();
	if (rmap_item) {
		/* It has already been zeroed */
		rmap_item->mm = mm_slot->mm;
		rmap_item->address = addr;
		rmap_item->rmap_list = *rmap_list;
#ifdef CONFIG_LKSM_FILTER
		rmap_item->region = ksm_scan.region;
#endif
		*rmap_list = rmap_item;
		if (lksm_test_mm_state(mm_slot, FROZEN_BIT))
			lksm_set_rmap_frozen(rmap_item);
		else
			lksm_clear_rmap_frozen(rmap_item);
	}
	return rmap_item;
}

/*
 * lksm_flush_removed_mm_list:
 * batched flushing out removed mm_slots by lksm_remove_mm_slot
 */
static void lksm_flush_removed_mm_list(void)
{
	struct mm_slot *head, *next, *slot;

	spin_lock(&ksm_mmlist_lock);
	head = list_first_entry_or_null(&ksm_scan.remove_mm_list,
			struct mm_slot, mm_list);
	if (!head) {
		spin_unlock(&ksm_mmlist_lock);
		return;
	}

	list_del_init(&ksm_scan.remove_mm_list);
	spin_unlock(&ksm_mmlist_lock);

	if (!list_empty(&head->mm_list)) {
		list_for_each_entry_safe(slot, next, &head->mm_list, mm_list) {
			list_del(&slot->mm_list);

			cond_resched();

			remove_trailing_rmap_items(&slot->rmap_list);
#ifdef CONFIG_LKSM_FILTER
			lksm_region_ref_list_release(slot);
#endif
			clear_bit(MMF_VM_MERGEABLE, &slot->mm->flags);
			mmdrop(slot->mm);
			free_mm_slot(slot);
		}
	}

	cond_resched();
	remove_trailing_rmap_items(&head->rmap_list);
#ifdef CONFIG_LKSM_FILTER
	lksm_region_ref_list_release(head);
#endif
	clear_bit(MMF_VM_MERGEABLE, &head->mm->flags);
	mmdrop(head->mm);
	free_mm_slot(head);
}

/*
 * remove mm_slot from lists
 * LKSM defers releasing stuffs at the end of scanning
 */
static inline void lksm_remove_mm_slot(struct mm_slot *slot)
{
	hash_del(&slot->link);
	list_del_init(&slot->scan_list);
	list_move(&slot->mm_list, &ksm_scan.remove_mm_list);
	if (!RB_EMPTY_NODE(&slot->ordered_list)) {
		rb_erase(&slot->ordered_list, &vips_list);
		RB_CLEAR_NODE(&slot->ordered_list);
	}
}

/* caller must hold ksm_mmlist_lock */
static struct mm_slot *lksm_get_unscanned_mm_slot(struct mm_slot *slot)
{
	struct mm_slot *next;

	list_for_each_entry_safe_continue(slot, next, &ksm_scan_head.scan_list,
			scan_list) {
		if (ksm_test_exit(slot->mm)) {
			if (lksm_test_mm_state(slot, KSM_MM_FROZEN))
				atomic_dec(&ksm_scan.nr_frozen);
			else
				atomic_dec(&ksm_scan.nr_scannable);
			lksm_remove_mm_slot(slot);
			continue;
		}

		lksm_nr_scanned_slot++;
		break;
	}

	return slot;
}

/* caller must hold ksm_mmlist_lock */
static void lksm_insert_mm_slot_ordered(struct mm_slot *slot)
{
	struct rb_root *root;
	struct rb_node **new;
	struct rb_node *parent;
	struct mm_slot *temp_slot;

	root = &vips_list;
	parent = NULL;
	new = &root->rb_node;

	while (*new) {
		temp_slot = rb_entry(*new, struct mm_slot, ordered_list);

		parent = *new;
		if (slot->nr_merged > temp_slot->nr_merged)
			new = &parent->rb_left;
		else
			new = &parent->rb_right;
	}

	rb_link_node(&slot->ordered_list, parent, new);
	rb_insert_color(&slot->ordered_list, root);
}

#ifdef CONFIG_LKSM_FILTER
/*
 * most vmas grow up except stack.
 * the given value of size must be same with orig's one.
 */

static inline void __lksm_copy_filter
(unsigned long *orig, unsigned long *newer, unsigned long size)
{
	while (size-- > 0)
		*(newer++) = *(orig++);
}

static inline void lksm_copy_filter
(struct lksm_region *region, unsigned long *filter)
{
	if (region->len > SINGLE_FILTER_LEN) {
		if (region->filter)
			__lksm_copy_filter(region->filter, filter, region->len);
	} else
		__lksm_copy_filter(&region->single_filter, filter, region->len);
}

static struct vm_area_struct *lksm_find_next_vma
(struct mm_struct *mm, struct mm_slot *slot)
{
	struct vm_area_struct *vma;
	struct lksm_region *region;

	if (ksm_test_exit(mm))
		vma = NULL;
	else
		vma = find_vma(mm, ksm_scan.address);
	for (; vma; vma = vma->vm_next) {
		if (!(vma->vm_flags & VM_MERGEABLE))
			continue;
		if (ksm_scan.address < vma->vm_start)
			ksm_scan.address = vma->vm_start;
		if (!vma->anon_vma) {
			ksm_scan.address = vma->vm_end;
			continue;
		}

		if (ksm_scan.cached_vma == vma)
			region = ksm_scan.region;
		else {
			region = lksm_find_region(vma);
			ksm_scan.cached_vma = vma;
			ksm_scan.vma_base_addr = vma->vm_start;
		}

		if (!region || region->type == LKSM_REGION_CONFLICT)
			region = &unknown_region;
		else if (region->type != LKSM_REGION_HEAP
					&& region->type != LKSM_REGION_CONFLICT
					&& region->type != LKSM_REGION_UNKNOWN) {
			unsigned long size = lksm_region_size(vma->vm_start, vma->vm_end);
			unsigned long len = (size > BITS_PER_LONG) ? lksm_bitmap_size(size)
					: SINGLE_FILTER_LEN;

			if (len > SINGLE_FILTER_LEN && unlikely(region->len != len)) {
				region->conflict++;
				if (region->conflict > 1) {
					region->type = LKSM_REGION_CONFLICT;
					if (region->len > SINGLE_FILTER_LEN)
						kfree(region->filter);
					region->filter = NULL;
					region->len = SINGLE_FILTER_LEN;
					/* conflicted regions will be unfiltered */
					region = &unknown_region;
					break;
				}
				if (region->len < len) {
					unsigned long *filter;
					ksm_debug("size of region(%p) is changed: %lu -> %lu (size: %lu)",
							region, region->len, len, size);
					filter = kcalloc(len, sizeof(long), GFP_KERNEL);
					if (!filter) {
						ksm_err("fail to allocate memory for filter");
						goto skip;
					}
					if (region->filter_cnt > 0)
						lksm_copy_filter(region, filter);
					if (region->len > SINGLE_FILTER_LEN)
						kfree(region->filter);
					region->filter = filter;
					region->len = len;
				}
			}
		}
skip:
		if (ksm_scan.region != region)
			ksm_scan.region = region;
		break;
	}
	return vma;
}

static inline unsigned long lksm_get_next_filtered_address
(struct lksm_region *region, unsigned long addr, unsigned long base)
{
	unsigned long next_offset, curr_offset, nbits;

	curr_offset = (addr - base) >> PAGE_SHIFT;
	nbits = region->len * BITS_PER_LONG;

	if (region->len > SINGLE_FILTER_LEN)
		next_offset = find_next_bit(region->filter, nbits, curr_offset);
	else
		next_offset = find_next_bit(&region->single_filter,
				nbits, curr_offset);

	return (next_offset << PAGE_SHIFT) + base;
}

#define lksm_region_skipped(region) \
		(region->len > 0 && !region->filter)

static struct rmap_item *__scan_next_rmap_item(struct page **page,
		struct mm_struct *mm, struct mm_slot *slot)
{
	struct vm_area_struct *vma;
	struct rmap_item *rmap_item;
	unsigned long addr;

again:
	cond_resched();
	vma = lksm_find_next_vma(mm, slot);

	while (vma && ksm_scan.address < vma->vm_end) {
		if (ksm_test_exit(mm)) {
			vma = NULL;
			break;
		}
		if (!lksm_test_mm_state(slot, KSM_MM_NEWCOMER)
				&& !lksm_test_mm_state(slot, KSM_MM_FROZEN)
				&& ksm_scan.region->type != LKSM_REGION_HEAP
				&& ksm_scan.region->type != LKSM_REGION_UNKNOWN
				&& lksm_region_mature(ksm_scan.scan_round, ksm_scan.region)
				&& !lksm_region_skipped(ksm_scan.region)) {
			if (ksm_scan.region->filter_cnt > 0) {
				addr = lksm_get_next_filtered_address(ksm_scan.region,
						ksm_scan.address, ksm_scan.vma_base_addr);
				ksm_scan.address = addr;
				if (ksm_scan.address >= vma->vm_end)
					break;
				if (ksm_scan.address < vma->vm_start) {
					ksm_debug("address(%lu) is less than vm_start(%lu)",
						ksm_scan.address, vma->vm_start);
					break;
				}
			} else {
				ksm_scan.address = vma->vm_end;
				break;
			}
		}
		*page = follow_page(vma, ksm_scan.address, FOLL_GET);
		if (IS_ERR_OR_NULL(*page)) {
			ksm_scan.address += PAGE_SIZE;
			cond_resched();
			continue;
		}
		if (PageAnon(*page)) {
			flush_anon_page(vma, *page, ksm_scan.address);
			flush_dcache_page(*page);
			rmap_item = get_next_rmap_item(slot,
					ksm_scan.rmap_list, ksm_scan.address);
			if (rmap_item) {
				ksm_scan.rmap_list =
						&rmap_item->rmap_list;
				ksm_scan.address += PAGE_SIZE;
			} else
				put_page(*page);
			return rmap_item;
		}
		put_page(*page);
		ksm_scan.address += PAGE_SIZE;
		cond_resched();
	}
	if (vma)
		goto again;
	/* clean-up a scanned region */
	ksm_scan.region = NULL;
	ksm_scan.cached_vma = NULL;
	ksm_scan.vma_base_addr = 0;

	return NULL; /* no scannable rmap item */
}

#else /* CONFIG_LKSM_FILTER */

static struct rmap_item *__scan_next_rmap_item(struct page **page,
		struct mm_struct *mm, struct mm_slot *slot)
{
	struct vm_area_struct *vma;
	struct rmap_item *rmap_item;

	if (ksm_test_exit(mm))
		vma = NULL;
	else
		vma = find_vma(mm, ksm_scan.address);

	for (; vma; vma = vma->vm_next) {
		if (!(vma->vm_flags & VM_MERGEABLE))
			continue;
		if (ksm_scan.address < vma->vm_start)
			ksm_scan.address = vma->vm_start;
		if (!vma->anon_vma)
			ksm_scan.address = vma->vm_end;

		while (ksm_scan.address < vma->vm_end) {
			if (ksm_test_exit(mm))
				break;
			*page = follow_page(vma, ksm_scan.address, FOLL_GET);
			if (IS_ERR_OR_NULL(*page)) {
				ksm_scan.address += PAGE_SIZE;
				cond_resched();
				continue;
			}
			if (PageAnon(*page)) {
				flush_anon_page(vma, *page, ksm_scan.address);
				flush_dcache_page(*page);
				rmap_item = get_next_rmap_item(slot,
					ksm_scan.rmap_list, ksm_scan.address);
				if (rmap_item) {
					ksm_scan.rmap_list =
							&rmap_item->rmap_list;
					ksm_scan.address += PAGE_SIZE;
				} else
					put_page(*page);
				return rmap_item;
			}
			put_page(*page);
			ksm_scan.address += PAGE_SIZE;
			cond_resched();
		}
	}

	return NULL;
}

#endif /* CONFIG_LKSM_FILTER */

static inline int sum_merge_win(int merge_win[], int len)
{
	int i, sum = 0;

	for (i = 0; i < len; i++)
		sum += merge_win[i];
	return sum;
}

static inline int lksm_account_mm_slot_nr_merge(struct mm_slot *slot, int nr_merged)
{
	slot->nr_merged_win[slot->merge_idx++] = nr_merged;
	if (slot->merge_idx == MERGE_WIN)
		slot->merge_idx = 0;
	slot->nr_merged = sum_merge_win(slot->nr_merged_win, MERGE_WIN);
	return slot->nr_merged;
}

static struct rmap_item *scan_get_next_rmap_item(struct page **page)
{
	struct mm_struct *mm;
	struct mm_slot *slot;
	struct rmap_item *rmap_item;

	if (list_empty(&ksm_scan_head.scan_list))
		return NULL;

	slot = ksm_scan.mm_slot;
	if (slot == &ksm_scan_head) {
		/*
		 * A number of pages can hang around indefinitely on per-cpu
		 * pagevecs, raised page count preventing write_protect_page
		 * from merging them.  Though it doesn't really matter much,
		 * it is puzzling to see some stuck in pages_volatile until
		 * other activity jostles them out, and they also prevented
		 * LTP's KSM test from succeeding deterministically; so drain
		 * them here (here rather than on entry to ksm_do_scan(),
		 * so we don't IPI too often when pages_to_scan is set low).
		 */
		lru_add_drain_all();

		if (ksm_scan.scan_round < ksm_crawl_round) {
			ksm_scan.scan_round = ksm_crawl_round;
			root_unstable_tree[LKSM_NODE_ID] = RB_ROOT;
		}

		spin_lock(&ksm_mmlist_lock);
		slot = lksm_get_unscanned_mm_slot(slot);
		ksm_scan.mm_slot = slot;
		spin_unlock(&ksm_mmlist_lock);

		/*
		 * Although we tested list_empty() above, a racing __ksm_exit
		 * of the last mm on the list may have removed it since then.
		 */
		if (slot == &ksm_scan_head)
			return NULL;

		slot->elapsed = get_jiffies_64();
next_mm:
		ksm_scan.address = 0;
		ksm_scan.rmap_list = &slot->rmap_list;
	}

	if (unlikely(!ksm_scan.rmap_list))
		ksm_scan.rmap_list = &slot->rmap_list;

	mm = slot->mm;
	BUG_ON(!mm);
	down_read(&mm->mmap_lock);
	rmap_item = __scan_next_rmap_item(page, mm, slot);

	if (rmap_item) {
		slot->nr_scans++;
		up_read(&mm->mmap_lock);
		return rmap_item;
	}

	if (ksm_test_exit(mm)) {
		ksm_scan.address = 0;
		ksm_scan.rmap_list = &slot->rmap_list;
	}
	/*
	 * Nuke all the rmap_items that are above this current rmap:
	 * because there were no VM_MERGEABLE vmas with such addresses.
	 */
	remove_trailing_rmap_items(ksm_scan.rmap_list);

	spin_lock(&ksm_mmlist_lock);
	ksm_scan.mm_slot = lksm_get_unscanned_mm_slot(slot);

	if (ksm_scan.address == 0) {
		/*
		 * We've completed a full scan of all vmas, holding mmap_lock
		 * throughout, and found no VM_MERGEABLE: so do the same as
		 * __ksm_exit does to remove this mm from all our lists now.
		 * This applies either when cleaning up after __ksm_exit
		 * (but beware: we can reach here even before __ksm_exit),
		 * or when all VM_MERGEABLE areas have been unmapped (and
		 * mmap_lock then protects against race with MADV_MERGEABLE).
		 */
		up_read(&mm->mmap_lock);
		if (lksm_test_mm_state(slot, KSM_MM_FROZEN))
			atomic_dec(&ksm_scan.nr_frozen);
		else
			atomic_dec(&ksm_scan.nr_scannable);
		lksm_remove_mm_slot(slot);
		spin_unlock(&ksm_mmlist_lock);

		lksm_slot_nr_merged = 0;
		lksm_slot_nr_broken = 0;
	} else {
		int newcomer = 0, frozen = 0;

		up_read(&mm->mmap_lock);

		if (lksm_test_mm_state(slot, KSM_MM_NEWCOMER)) {
			lksm_clear_mm_state(slot, KSM_MM_NEWCOMER);
			newcomer = 1;
		}
		if (lksm_test_mm_state(slot, KSM_MM_FROZEN)) {
			lksm_clear_mm_state(slot, KSM_MM_FROZEN);
			frozen = 1;
			atomic_dec(&ksm_scan.nr_frozen);
		} else
			atomic_dec(&ksm_scan.nr_scannable);
		lksm_set_mm_state(slot, KSM_MM_SCANNED);

		list_del_init(&slot->scan_list);
		if (!RB_EMPTY_NODE(&slot->ordered_list)) {
			rb_erase(&slot->ordered_list, &vips_list);
			RB_CLEAR_NODE(&slot->ordered_list);
		}
		if (lksm_account_mm_slot_nr_merge(slot, lksm_slot_nr_merged))
			lksm_insert_mm_slot_ordered(slot);

		slot->elapsed = get_jiffies_64() - slot->elapsed;
		spin_unlock(&ksm_mmlist_lock);

		if (ksm_test_exit(slot->mm))
			ksm_debug("slot(%p:%p) is exited", slot, slot->mm);
		else
			ksm_debug("slot-%d(%s) %d merged %d scanned %lu pages "
					"(sum: %d) - (%s, %s) takes %u msecs (nr_scannable: %d)",
					task_pid_nr(slot->mm->owner), slot->mm->owner->comm,
					lksm_slot_nr_merged - lksm_slot_nr_broken, slot->nr_scans,
					slot->scanning_size, slot->nr_merged,
					newcomer ? "new" : "old",
					frozen ? "frozen" : "normal",
					jiffies_to_msecs(slot->elapsed),
					atomic_read(&ksm_scan.nr_scannable));

		lksm_slot_nr_merged = 0;
		lksm_slot_nr_broken = 0;
	}

	/* Repeat until we've completed scanning the whole list */
	slot = ksm_scan.mm_slot;
	if (slot != &ksm_scan_head) {
		slot->elapsed = get_jiffies_64();
		goto next_mm;
	}

	return NULL;
}

/**
 * ksm_do_scan  - the ksm scanner main worker function.
 * @scan_npages:  number of pages we want to scan before we return.
 */
static int ksm_do_scan(unsigned int scan_npages)
{
	struct rmap_item *rmap_item;
	struct page *page;

	while (scan_npages-- && likely(!freezing(current))) {
		cond_resched();
		rmap_item = scan_get_next_rmap_item(&page);
		if (!rmap_item)
			return 1; /* need sleep */
		cmp_and_merge_page(page, rmap_item);
		put_page(page);
	}
	return 0;
}

static int ksmd_should_run(void)
{
	return (ksm_run & KSM_RUN_MERGE) &&
		!list_empty(&ksm_scan_head.scan_list);
}

static void lksm_scan_wrapup_wait(void)
{
	if (ksm_scan.scan_mode == LKSM_SCAN_PARTIAL) {
		if (ksm_thread_pages_to_scan != lksm_default_pages_to_scan)
			ksm_thread_pages_to_scan = lksm_default_pages_to_scan;
	} else if (ksm_scan.scan_mode == LKSM_SCAN_FULL)
		ksm_scan.nr_full_scan++;

	lksm_nr_merged = 0;
	lksm_nr_broken = 0;
	lksm_nr_scanned_slot = 0;

	ksm_scan.scan_mode = LKSM_SCAN_NONE;
	if (ksm_run & KSM_RUN_ONESHOT)
		atomic_set(&ksm_one_shot_scanning, LKSM_SCAN_NONE);

	lksm_clear_scan_state(ksm_state);

	wait_event_freezable(ksm_thread_wait,
			(lksm_check_scan_state(ksm_state) && ksmd_should_run())
			|| kthread_should_stop());
}

static int lksm_scan_thread(void *nothing)
{
	unsigned long begin, elapsed;
	unsigned int sleep_ms;
	int need_to_sleep = 0;

	set_freezable();
	set_user_nice(current, 5);

	ksm_debug("KSM_SCAND pid: %d", task_pid_nr(current));
	while (!kthread_should_stop()) {
		mutex_lock(&ksm_thread_mutex);
		wait_while_offlining();
		if (ksmd_should_run())
			need_to_sleep = ksm_do_scan(ksm_thread_pages_to_scan);
		mutex_unlock(&ksm_thread_mutex);

		try_to_freeze();

		if (need_to_sleep) {
			if (!ksmd_should_run()) {
				/* if no one left in scanning list, go to sleep for a while */
				lksm_flush_removed_mm_list();

				elapsed = get_jiffies_64() - begin;
				lksm_last_scan_time = elapsed;
				lksm_proc_scan_time = elapsed / lksm_nr_scanned_slot;

				ksm_debug("Scanning(%d) takes %u ms, %d/%d-pages "
						"are merged/broken (nr_scannable: %d, nr_frozen: %d)",
						lksm_nr_scanned_slot,
						jiffies_to_msecs(lksm_last_scan_time),
						lksm_nr_merged, lksm_nr_broken,
						atomic_read(&ksm_scan.nr_scannable),
						atomic_read(&ksm_scan.nr_frozen));

				lksm_scan_wrapup_wait();

				ksm_debug("Start %lu-th scanning: nr_scannable(%d) "
						"nr_frozen(%d)",
						ksm_scan.scan_round,
						atomic_read(&ksm_scan.nr_scannable),
						atomic_read(&ksm_scan.nr_frozen));

				if (ksm_scan.scan_mode == LKSM_SCAN_PARTIAL) {
					if (lksm_boosted_pages_to_scan !=
							ksm_thread_pages_to_scan) {
						ksm_thread_pages_to_scan = lksm_boosted_pages_to_scan;
						ksm_debug("set pages_to_scan to %u",
								lksm_boosted_pages_to_scan);
					}
				}
				begin = get_jiffies_64();
			} else {
				/* new scanning targets are coming */
				sleep_ms = READ_ONCE(ksm_thread_sleep_millisecs);
				wait_event_interruptible_timeout(ksm_iter_wait,
						sleep_ms != READ_ONCE(ksm_thread_sleep_millisecs),
						msecs_to_jiffies(sleep_ms));
			}
			need_to_sleep = 0;
		} else if (ksmd_should_run()) {
			/* normal sleep */
			sleep_ms = READ_ONCE(ksm_thread_sleep_millisecs);
			wait_event_interruptible_timeout(ksm_iter_wait,
				sleep_ms != READ_ONCE(ksm_thread_sleep_millisecs),
				msecs_to_jiffies(sleep_ms));
		} else {
			/* wait for activating ksm */
			if (likely(ksm_scan.scan_round > 0)) {
				lksm_flush_removed_mm_list();

				elapsed = get_jiffies_64() - begin;
				lksm_last_scan_time = elapsed;
				lksm_proc_scan_time = elapsed / lksm_nr_scanned_slot;

				ksm_debug("Scanning(%d) takes %u ms, %d/%d-pages are merged/broken",
						lksm_nr_scanned_slot, jiffies_to_msecs(lksm_last_scan_time),
						lksm_nr_merged, lksm_nr_broken);

				lksm_scan_wrapup_wait();
			} else
				wait_event_freezable(ksm_thread_wait,
					(lksm_check_scan_state(ksm_state) && ksmd_should_run())
					|| kthread_should_stop());

			ksm_debug("Start %lu-th scanning: nr_scannable(%d) nr_frozen(%d)",
					ksm_scan.scan_round,
					atomic_read(&ksm_scan.nr_scannable),
					atomic_read(&ksm_scan.nr_frozen));

			if (ksm_scan.scan_mode == LKSM_SCAN_PARTIAL) {
				ksm_thread_pages_to_scan = lksm_boosted_pages_to_scan;
				ksm_debug("set pages_to_scan to %u",
						lksm_boosted_pages_to_scan);
			}
			begin = get_jiffies_64();
		}
	}
	return 0;
}

/*
 * lksm crawler declaration & definition part
 */
static struct task_struct *ksm_crawld;

LIST_HEAD(frozen_task_list);
DEFINE_SPINLOCK(frozen_task_lock);

enum {
	KSM_CRAWL_SLEEP,
	KSM_CRAWL_RUN,
} ksm_crawl_state;
static atomic_t crawl_state;

enum {
	LKSM_TASK_SLOT_NONE = 0,
	LKSM_TASK_SLOT_REMOVED,
};

static inline int lksm_count_and_clear_mm_slots
(struct mm_slot *head, unsigned long *delay)
{
	int count = 0;
	struct mm_slot *slot;

	spin_lock(&ksm_mmlist_lock);
	list_for_each_entry(slot, &head->mm_list, mm_list) {
		if (list_empty(&slot->scan_list)) {
			lksm_clear_mm_state(slot, KSM_MM_SCANNED);
			slot->nr_scans = 0;
			slot->scanning_size = get_mm_counter(slot->mm, MM_ANONPAGES);
			list_add_tail(&slot->scan_list, &ksm_scan_head.scan_list);
			*delay += slot->elapsed;
			count++;
		}
	}
	spin_unlock(&ksm_mmlist_lock);
	return count;
}

static int lksm_prepare_frozen_scan(void)
{
	int nr_frozen = 0, nr_added = 0, err;
	struct task_struct *task;
	struct task_slot *task_slot;
	struct mm_struct *mm;

	spin_lock(&frozen_task_lock);
	nr_frozen = atomic_read(&ksm_scan.nr_frozen);
	if (list_empty(&frozen_task_list)) {
		spin_unlock(&frozen_task_lock);
		return nr_frozen;
	}

	ksm_debug("prepare frozen scan: round(%lu)", ksm_crawl_round);
	task_slot = list_first_entry_or_null(&frozen_task_list,
			struct task_slot, list);
	while (task_slot) {
		list_del(&task_slot->list);
		hash_del(&task_slot->hlist);
		spin_unlock(&frozen_task_lock);

		task = task_slot->task;
		if (ksm_run & KSM_RUN_UNMERGE) {
			put_task_struct(task);
			free_task_slot(task_slot);
			goto clean_up_abort;
		}

		mm = get_task_mm(task);

		if (!mm || ksm_test_exit(mm))
			goto mm_exit_out;

		if (mm) {
			ksm_join_write_lock(mm, task_slot->frozen, err);
			if (!err)
				nr_added++;
		}

mm_exit_out:
		free_task_slot(task_slot);
		put_task_struct(task);
		if (mm)
			mmput(mm);

		cond_resched();

		spin_lock(&frozen_task_lock);
		task_slot = list_first_entry_or_null(&frozen_task_list,
				struct task_slot, list);
	}
	spin_unlock(&frozen_task_lock);
	atomic_add(nr_added, &ksm_scan.nr_frozen);

	return nr_added + nr_frozen;

clean_up_abort:
	spin_lock(&frozen_task_lock);
	task_slot = list_first_entry_or_null(&frozen_task_list,
			struct task_slot, list);
	while (task_slot) {
		list_del(&task_slot->list);
		hash_del(&task_slot->hlist);
		spin_unlock(&frozen_task_lock);

		task = task_slot->task;
		put_task_struct(task);
		free_task_slot(task_slot);

		spin_lock(&frozen_task_lock);
		task_slot = list_first_entry_or_null(&frozen_task_list,
				struct task_slot, list);
	}
	spin_unlock(&frozen_task_lock);

	return 0;
}

/* this function make a list of new processes and vip processes */
static int lksm_prepare_partial_scan(void)
{
	int ret, nr_frozen = 0, nr_added = 0, nr_scannable = 0;
	unsigned long delay = 0;
	unsigned long fault_cnt = 0;
	struct task_struct *task;
	struct mm_struct *mm;
	struct mm_slot *mm_slot;
	struct list_head recheck_list;
	struct rb_node *node;

	ksm_debug("prepare partial scan: round(%lu)", ksm_crawl_round);
	INIT_LIST_HEAD(&recheck_list);

	nr_frozen = lksm_prepare_frozen_scan();

	/* get newbies */
	for_each_process(task) {
		if (task == current || task_pid_nr(task) == 0
			|| check_short_task(task))
			continue;
		if (ksm_run & KSM_RUN_UNMERGE) {
			nr_frozen = 0;
			nr_added = 0;
			goto abort;
		}
		mm = get_task_mm(task);
		if (!mm)
			continue;
		ksm_join_write_lock(mm, KSM_TASK_UNFROZEN, ret);
		if (ret > 0)
			nr_added++;
		mmput(mm);
	}

	/* get vips */
	if (nr_added + nr_frozen >= lksm_max_vips) {
		ksm_debug("nr_scannable(%d) already fulfilled skip vips",
				nr_added + nr_frozen);
		goto skip_vips;
	}

	spin_lock(&ksm_mmlist_lock);
	node = rb_first(&vips_list);
	if (!node) {
		ksm_debug("empty vip list");
		spin_unlock(&ksm_mmlist_lock);
		goto skip_vips;
	}
	mm_slot = rb_entry(node, struct mm_slot, ordered_list);
	while (nr_scannable + nr_added + nr_frozen < lksm_max_vips) {
		if (ksm_run & KSM_RUN_UNMERGE) {
			spin_unlock(&ksm_mmlist_lock);
			nr_scannable = 0;
			nr_frozen = 0;
			nr_added = 0;
			goto abort;
		}
		if (ksm_test_exit(mm_slot->mm)) {
			if (!lksm_test_mm_state(mm_slot, KSM_MM_SCANNED))
				atomic_dec(&ksm_scan.nr_scannable);
			lksm_remove_mm_slot(mm_slot);
			goto next_node;
		}
		if (!lksm_test_mm_state(mm_slot, KSM_MM_LISTED))
			goto next_node;

		/* prunning by fault count */
		fault_cnt = mm_slot->mm->owner->maj_flt + mm_slot->mm->owner->min_flt;
		if (mm_slot->fault_cnt == fault_cnt)
			goto next_node;

		mm_slot->fault_cnt = fault_cnt;
		mm_slot->scanning_size = get_mm_counter(mm_slot->mm, MM_ANONPAGES);
		mm_slot->nr_scans = 0;
		delay += mm_slot->elapsed;
		ksm_debug("slot(nr_merged: %d, scanning_size: %lu) task(%s)",
				mm_slot->nr_merged, mm_slot->scanning_size,
				mm_slot->mm->owner->comm);
		list_move_tail(&mm_slot->scan_list, &recheck_list);
		lksm_clear_mm_state(mm_slot, KSM_MM_SCANNED);
#ifdef CONFIG_LKSM_FILTER
		/* to prevent mm_slot termination on __ksm_exit */
		lksm_set_mm_state(mm_slot, KSM_MM_PREPARED);
#endif
		nr_scannable++;

next_node:
		node = rb_next(node);
		if (!node)
			break;
		mm_slot = rb_entry(node, struct mm_slot, ordered_list);
	}
	spin_unlock(&ksm_mmlist_lock);
#ifdef CONFIG_LKSM_FILTER
	list_for_each_entry(mm_slot, &recheck_list, scan_list) {
		if (ksm_test_exit(mm_slot->mm))
			continue;
		mm_slot->nr_scans = 0;
		/* check new maps */
		down_read(&mm_slot->mm->mmap_lock);
		ksm_join(mm_slot->mm, KSM_TASK_UNFROZEN);
		up_read(&mm_slot->mm->mmap_lock);
	}
#endif
skip_vips:
	spin_lock(&ksm_mmlist_lock);
	if (!list_empty(&recheck_list)) {
#ifdef CONFIG_LKSM_FILTER
        list_for_each_entry(mm_slot, &recheck_list, scan_list)
            lksm_clear_mm_state(mm_slot, KSM_MM_PREPARED);
#endif
		list_splice(&recheck_list, &ksm_scan_head.scan_list);
	}
	spin_unlock(&ksm_mmlist_lock);

	ksm_scan.scan_mode = LKSM_SCAN_PARTIAL;
	ksm_crawl_round++;

	atomic_add(nr_scannable + nr_added, &ksm_scan.nr_scannable);
	ksm_debug("nr_frozen: %d nr_added: %d nr_scannable: %d - %d",
		nr_frozen, nr_added, nr_scannable, atomic_read(&ksm_scan.nr_scannable));
abort:
	return nr_frozen + nr_added + nr_scannable;
}

static int lksm_prepare_full_scan(unsigned long *next_fullscan)
{
	int ret, nr_frozen = 0, nr_added = 0, nr_scannable = 0, nr_target;
	unsigned long delay = 0;
	struct task_struct *task;
	struct mm_struct *mm;

	ksm_debug("prepare full scan: round(%lu)", ksm_crawl_round);

	nr_frozen = lksm_prepare_frozen_scan();

	for_each_process(task) {
		if (task == current || task_pid_nr(task) == 0
			|| check_short_task(task))
			continue;
		if (ksm_run & KSM_RUN_UNMERGE) {
			nr_target = 0;
			goto abort;
		}

		mm = get_task_mm(task);
		if (!mm)
			continue;
		ksm_join_write_lock(mm, KSM_TASK_UNFROZEN, ret);
		if (ret > 0)
			nr_added++;
		mmput(mm);
	}

	nr_scannable = lksm_count_and_clear_mm_slots(&ksm_mm_head, &delay);
	nr_target = nr_scannable + nr_added + nr_frozen;

	/* calculate crawler's sleep time */
	delay += msecs_to_jiffies((nr_frozen + nr_added) * lksm_proc_scan_time);
	*next_fullscan = jiffies + delay + msecs_to_jiffies(full_scan_interval);

	ksm_scan.scan_mode = LKSM_SCAN_FULL;
	ksm_crawl_round++;

	atomic_add(nr_scannable + nr_added, &ksm_scan.nr_scannable);
	ksm_debug("nr_frozen: %d nr_added: %d nr_scannable: %d - %d",
			nr_frozen, nr_added, nr_scannable,
			atomic_read(&ksm_scan.nr_scannable));
abort:
	return nr_target;
}

static int lksm_do_wait_userspace_event(unsigned long sleep_time)
{
	wait_event_freezable(ksm_crawl_wait,
			kthread_should_stop() ||
			(atomic_read(&ksm_one_shot_scanning) > 0));
	return atomic_read(&ksm_one_shot_scanning);
}

static int lksm_do_wait_frozen_event(unsigned long sleep_time)
{
	int need_scan = 0;

	spin_lock_irq(&frozen_task_lock);
	if (list_empty(&frozen_task_list))
		/* wait until candidate list is filled */
		wait_event_interruptible_lock_irq_timeout(
				ksm_crawl_wait,
				kthread_should_stop()
				|| !list_empty(&frozen_task_list)
				|| !list_empty(&ksm_scan_head.scan_list),
				frozen_task_lock, sleep_time);

	if (!list_empty(&frozen_task_list) ||
			!list_empty(&ksm_scan_head.scan_list))
		need_scan = 1;
	spin_unlock_irq(&frozen_task_lock);

	return need_scan;
}

static inline void lksm_wake_up_scan_thread(void)
{
	ksm_debug("wake up lksm_scan_thread");
	lksm_set_scan_state(ksm_state);
	wake_up(&ksm_thread_wait);
}

#define LKSM_CRAWL_FROZEN_EVENT_WAIT 100 /* 100ms */

static void lksm_do_crawl_once
(unsigned long *next_fscan, unsigned long sleep_time)
{
	int nr_added = 0;
	int scan_mode;

	/* cralwer thread waits for trigger event from userspace */
	scan_mode = lksm_do_wait_userspace_event(sleep_time);

	if (scan_mode == LKSM_SCAN_PARTIAL) {
		atomic_set(&crawl_state, KSM_CRAWL_RUN);
		msleep(LKSM_CRAWL_FROZEN_EVENT_WAIT);
		nr_added = lksm_prepare_partial_scan();
	} else if (scan_mode == LKSM_SCAN_FULL) {
		atomic_set(&crawl_state, KSM_CRAWL_RUN);
		nr_added = lksm_prepare_full_scan(next_fscan);
	}

	try_to_freeze();

	if (nr_added > 0)
		lksm_wake_up_scan_thread();
	else {
		ksm_debug("No one can be scanned!");
		atomic_set(&ksm_one_shot_scanning, LKSM_SCAN_NONE);
	}
	atomic_set(&crawl_state, KSM_CRAWL_SLEEP);
}

static void lksm_do_crawl_periodic
(unsigned long *next_fscan, unsigned long sleep_time)
{
	int nr_added = 0;

	if (time_is_before_eq_jiffies(*next_fscan)) {
		atomic_set(&crawl_state, KSM_CRAWL_RUN);
		nr_added = lksm_prepare_full_scan(next_fscan);
	} else if (lksm_do_wait_frozen_event(sleep_time)) {
		atomic_set(&crawl_state, KSM_CRAWL_RUN);
		msleep(LKSM_CRAWL_FROZEN_EVENT_WAIT);
		nr_added = lksm_prepare_partial_scan();
	}

	try_to_freeze();

	if (nr_added > 0)
		lksm_wake_up_scan_thread();
	atomic_set(&crawl_state, KSM_CRAWL_SLEEP);
}

static int lksm_crawl_thread(void *data)
{
	int nr_added = 0;
	unsigned long next_fscan = jiffies;	/* next full scan */
	unsigned long sleep_time = crawler_sleep;

	set_freezable();
	set_user_nice(current, 5);

	ksm_debug("KSM_CRAWLD pid: %d", task_pid_nr(current));
	wait_event_freezable(ksm_crawl_wait,
		kthread_should_stop() || ksm_run & KSM_RUN_MERGE);
	/* initial loop */
	while (!kthread_should_stop() && ksm_crawl_round < initial_round) {

		try_to_freeze();

		if ((ksm_run & KSM_RUN_MERGE) &&
				!lksm_check_scan_state(ksm_state) &&
				time_is_before_eq_jiffies(next_fscan)) {
			nr_added = lksm_prepare_full_scan(&next_fscan);
			if (nr_added) {
				lksm_wake_up_scan_thread();
				nr_added = 0;
			}
			next_fscan = jiffies + sleep_time;
		}

		wait_event_interruptible_timeout(ksm_crawl_wait,
			kthread_should_stop() || !lksm_check_scan_state(ksm_state),
			sleep_time);
	}

	/* initialization loop done */
	full_scan_interval = DEFAULT_FULL_SCAN_INTERVAL;
	next_fscan = jiffies + msecs_to_jiffies(full_scan_interval);
	atomic_set(&crawl_state, KSM_CRAWL_SLEEP);

	/* normal operation loop */
	while (!kthread_should_stop()) {
		if (ksm_run & KSM_RUN_ONESHOT) {
			if (!lksm_check_scan_state(ksm_state))
				lksm_do_crawl_once(&next_fscan, sleep_time);
			else
				/* wait until scanning done */
				wait_event_freezable(ksm_crawl_wait,
					!lksm_check_scan_state(ksm_state)
					|| kthread_should_stop());
		} else if (ksm_run & KSM_RUN_MERGE) {
			if (!lksm_check_scan_state(ksm_state))
				lksm_do_crawl_periodic(&next_fscan, sleep_time);
			else
				/* wait until scanning done */
				wait_event_interruptible_timeout(ksm_crawl_wait,
					!lksm_check_scan_state(ksm_state)
					|| kthread_should_stop(),
					sleep_time);
			try_to_freeze();
		} else {
			ksm_debug("ksm is not activated");
			wait_event_freezable(ksm_crawl_wait,
				kthread_should_stop() || (ksm_run & KSM_RUN_MERGE));
		}
	}

	return 0;
}

int ksm_madvise(struct vm_area_struct *vma, unsigned long start,
		unsigned long end, int advice, unsigned long *vm_flags)
{
	struct mm_struct *mm = vma->vm_mm;
	int err;

	switch (advice) {
	case MADV_MERGEABLE:
		/*
		 * Be somewhat over-protective for now!
		 */
		if (*vm_flags & (VM_MERGEABLE | VM_SHARED  | VM_MAYSHARE   |
				 VM_PFNMAP    | VM_IO      | VM_DONTEXPAND |
				 VM_HUGETLB | VM_MIXEDMAP))
			return 0;		/* just ignore the advice */

		if (vma_is_dax(vma))
			return 0;

#ifdef VM_SAO
		if (*vm_flags & VM_SAO)
			return 0;
#endif
#ifdef VM_SPARC_ADI
		if (*vm_flags & VM_SPARC_ADI)
			return 0;
#endif

		if (!test_bit(MMF_VM_MERGEABLE, &mm->flags)) {
			err = __ksm_enter(mm, KSM_TASK_UNFROZEN);
			if (err)
				return err;
		}

		*vm_flags |= VM_MERGEABLE;
		break;

	case MADV_UNMERGEABLE:
		if (!(*vm_flags & VM_MERGEABLE))
			return 0;		/* just ignore the advice */

		if (vma->anon_vma) {
			err = unmerge_ksm_pages(vma, start, end);
			if (err)
				return err;
		}

		*vm_flags &= ~VM_MERGEABLE;
		break;
	}

	return 0;
}

static struct mm_slot *__ksm_enter_alloc_slot(struct mm_struct *mm, int frozen)
{
	struct mm_slot *mm_slot;

	mm_slot = alloc_mm_slot();
	if (!mm_slot)
		return NULL;

	if (frozen == KSM_TASK_FROZEN)
		lksm_set_mm_state(mm_slot, KSM_MM_FROZEN | KSM_MM_NEWCOMER);
	else
		lksm_set_mm_state(mm_slot, KSM_MM_LISTED | KSM_MM_NEWCOMER);

	lksm_clear_mm_state(mm_slot, KSM_MM_SCANNED);
	RB_CLEAR_NODE(&mm_slot->ordered_list);
	mm_slot->fault_cnt = mm->owner->maj_flt + mm->owner->min_flt;
	mm_slot->scanning_size = get_mm_counter(mm, MM_ANONPAGES);

	spin_lock(&ksm_mmlist_lock);
	insert_to_mm_slots_hash(mm, mm_slot);
	/*
	 * When KSM_RUN_MERGE (or KSM_RUN_STOP),
	 * insert just behind the scanning cursor, to let the area settle
	 * down a little; when fork is followed by immediate exec, we don't
	 * want ksmd to waste time setting up and tearing down an rmap_list.
	 *
	 * But when KSM_RUN_UNMERGE, it's important to insert ahead of its
	 * scanning cursor, otherwise KSM pages in newly forked mms will be
	 * missed: then we might as well insert at the end of the list.
	 */
	if (ksm_run & KSM_RUN_UNMERGE)
		list_add_tail(&mm_slot->mm_list, &ksm_mm_head.mm_list);
	else {
		list_add_tail(&mm_slot->scan_list, &ksm_scan_head.scan_list);
		list_add_tail(&mm_slot->mm_list, &ksm_mm_head.mm_list);
	}
	ksm_nr_added_process++;
	spin_unlock(&ksm_mmlist_lock);
#ifdef CONFIG_LKSM_FILTER
	INIT_LIST_HEAD(&mm_slot->ref_list);
#endif
	set_bit(MMF_VM_MERGEABLE, &mm->flags);
	atomic_inc(&mm->mm_count);

	return mm_slot;
}

int __ksm_enter(struct mm_struct *mm, int frozen)
{
	if (!__ksm_enter_alloc_slot(mm, frozen))
		return -ENOMEM;
	return 0;
}

void __ksm_exit(struct mm_struct *mm)
{
	struct mm_slot *mm_slot;
	int easy_to_free = 0;

	/*
	 * This process is exiting: if it's straightforward (as is the
	 * case when ksmd was never running), free mm_slot immediately.
	 * But if it's at the cursor or has rmap_items linked to it, use
	 * mmap_lock to synchronize with any break_cows before pagetables
	 * are freed, and leave the mm_slot on the list for ksmd to free.
	 * Beware: ksm may already have noticed it exiting and freed the slot.
	 */

	spin_lock(&ksm_mmlist_lock);
	mm_slot = get_mm_slot(mm);
	if (!mm_slot) {
		spin_unlock(&ksm_mmlist_lock);
		return;
	}

	if (ksm_scan.mm_slot != mm_slot) {
#ifdef CONFIG_LKSM_FILTER
		if (lksm_test_mm_state(mm_slot, KSM_MM_PREPARED))
			goto deferring_free;
#endif
		if (!mm_slot->rmap_list) {
			hash_del(&mm_slot->link);
			list_del(&mm_slot->mm_list);
			list_del(&mm_slot->scan_list);
			if (!RB_EMPTY_NODE(&mm_slot->ordered_list)) {
				rb_erase(&mm_slot->ordered_list, &vips_list);
				RB_CLEAR_NODE(&mm_slot->ordered_list);
			}
			easy_to_free = 1;
		} else
			lksm_remove_mm_slot(mm_slot);
		if (lksm_test_mm_state(mm_slot, KSM_MM_FROZEN))
			atomic_dec(&ksm_scan.nr_frozen);
		else if (!lksm_test_mm_state(mm_slot, KSM_MM_SCANNED))
			atomic_dec(&ksm_scan.nr_scannable);
	}
#ifdef CONFIG_LKSM_FILTER
deferring_free:
#endif
	ksm_nr_added_process--;
	spin_unlock(&ksm_mmlist_lock);

	if (easy_to_free) {
#ifdef CONFIG_LKSM_FILTER
		lksm_region_ref_list_release(mm_slot);
#endif
		free_mm_slot(mm_slot);
		clear_bit(MMF_VM_MERGEABLE, &mm->flags);
		mmdrop(mm);
	} else if (mm_slot) {
		down_write(&mm->mmap_lock);
		up_write(&mm->mmap_lock);
	}
}

struct page *ksm_might_need_to_copy(struct page *page,
			struct vm_area_struct *vma, unsigned long address)
{
	struct anon_vma *anon_vma = page_anon_vma(page);
	struct page *new_page;

	if (PageKsm(page)) {
		if (page_stable_node(page) &&
		    !(ksm_run & KSM_RUN_UNMERGE))
			return page;	/* no need to copy it */
	} else if (!anon_vma) {
		return page;		/* no need to copy it */
	} else if (page->index == linear_page_index(vma, address) &&
		anon_vma->root == vma->anon_vma->root) {
		return page;		/* still no need to copy it */
	}
	if (!PageUptodate(page))
		return page;		/* let do_swap_page report the error */

	new_page = alloc_page_vma(GFP_HIGHUSER_MOVABLE, vma, address);
	if (new_page && mem_cgroup_charge(new_page, vma->vm_mm, GFP_KERNEL)) {
		put_page(new_page);
		new_page = NULL;
	}
	if (new_page) {
		copy_user_highpage(new_page, page, address, vma);

		SetPageDirty(new_page);
		__SetPageUptodate(new_page);
		__SetPageLocked(new_page);
#ifdef CONFIG_SWAP
		count_vm_event(KSM_SWPIN_COPY);
#endif
	}

	return new_page;
}

void rmap_walk_ksm(struct page *page, struct rmap_walk_control *rwc)
{
	struct stable_node *stable_node;
	struct rmap_item *rmap_item;
	int search_new_forks = 0;

	VM_BUG_ON_PAGE(!PageKsm(page), page);

	/*
	 * Rely on the page lock to protect against concurrent modifications
	 * to that page's node of the stable tree.
	 */
	VM_BUG_ON_PAGE(!PageLocked(page), page);

	stable_node = page_stable_node(page);
	if (!stable_node)
		return;
again:
	hlist_for_each_entry(rmap_item, &stable_node->hlist, hlist) {
		struct anon_vma *anon_vma = rmap_item->anon_vma;
		struct anon_vma_chain *vmac;
		struct vm_area_struct *vma;

		cond_resched();
		anon_vma_lock_read(anon_vma);
		anon_vma_interval_tree_foreach(vmac, &anon_vma->rb_root,
					       0, ULONG_MAX) {
			unsigned long addr;

			cond_resched();
			vma = vmac->vma;

			/* Ignore the stable/unstable/sqnr flags */
			addr = rmap_item->address & PAGE_MASK;

			if (addr < vma->vm_start || addr >= vma->vm_end)
				continue;
			/*
			 * Initially we examine only the vma which covers this
			 * rmap_item; but later, if there is still work to do,
			 * we examine covering vmas in other mms: in case they
			 * were forked from the original since ksmd passed.
			 */
			if ((rmap_item->mm == vma->vm_mm) == search_new_forks)
				continue;

			if (rwc->invalid_vma && rwc->invalid_vma(vma, rwc->arg))
				continue;

			if (!rwc->rmap_one(page, vma, addr, rwc->arg)) {
				anon_vma_unlock_read(anon_vma);
				return;
			}
			if (rwc->done && rwc->done(page)) {
				anon_vma_unlock_read(anon_vma);
				return;
			}
		}
		anon_vma_unlock_read(anon_vma);
	}
	if (!search_new_forks++)
		goto again;
}

#ifdef CONFIG_MIGRATION
void ksm_migrate_page(struct page *newpage, struct page *oldpage)
{
	struct stable_node *stable_node;

	VM_BUG_ON_PAGE(!PageLocked(oldpage), oldpage);
	VM_BUG_ON_PAGE(!PageLocked(newpage), newpage);
	VM_BUG_ON_PAGE(newpage->mapping != oldpage->mapping, newpage);

	stable_node = page_stable_node(newpage);
	if (stable_node) {
		VM_BUG_ON_PAGE(stable_node->kpfn != page_to_pfn(oldpage), oldpage);
		stable_node->kpfn = page_to_pfn(newpage);
		/*
		 * newpage->mapping was set in advance; now we need smp_wmb()
		 * to make sure that the new stable_node->kpfn is visible
		 * to get_ksm_page() before it can see that oldpage->mapping
		 * has gone stale (or that PageSwapCache has been cleared).
		 */
		smp_wmb();
		set_page_stable_node(oldpage, NULL);
	}
}
#endif /* CONFIG_MIGRATION */

#ifdef CONFIG_MEMORY_HOTREMOVE
static void wait_while_offlining(void)
{
	while (ksm_run & KSM_RUN_OFFLINE) {
		mutex_unlock(&ksm_thread_mutex);
		wait_on_bit(&ksm_run, ilog2(KSM_RUN_OFFLINE),
			    TASK_UNINTERRUPTIBLE);
		mutex_lock(&ksm_thread_mutex);
	}
}

static bool stable_node_dup_remove_range(struct stable_node *stable_node,
					 unsigned long start_pfn,
					 unsigned long end_pfn)
{
	if (stable_node->kpfn >= start_pfn &&
	    stable_node->kpfn < end_pfn) {
		/*
		 * Don't get_ksm_page, page has already gone:
		 * which is why we keep kpfn instead of page*
		 */
		remove_node_from_stable_tree(stable_node);
		return true;
	}
	return false;
}

static bool stable_node_chain_remove_range(struct stable_node *stable_node,
					   unsigned long start_pfn,
					   unsigned long end_pfn,
					   struct rb_root *root)
{
	struct stable_node *dup;
	struct hlist_node *hlist_safe;

	if (!is_stable_node_chain(stable_node)) {
		VM_BUG_ON(is_stable_node_dup(stable_node));
		return stable_node_dup_remove_range(stable_node, start_pfn,
						    end_pfn);
	}

	hlist_for_each_entry_safe(dup, hlist_safe,
				  &stable_node->hlist, hlist_dup) {
		VM_BUG_ON(!is_stable_node_dup(dup));
		stable_node_dup_remove_range(dup, start_pfn, end_pfn);
	}
	if (hlist_empty(&stable_node->hlist)) {
		free_stable_node_chain(stable_node, root);
		return true; /* notify caller that tree was rebalanced */
	} else
		return false;
}

static void ksm_check_stable_tree(unsigned long start_pfn,
				  unsigned long end_pfn)
{
	struct stable_node *stable_node, *next;
	struct rb_node *node;
	int nid;

	for (nid = 0; nid < ksm_nr_node_ids; nid++) {
		node = rb_first(root_stable_tree + nid);
		while (node) {
			stable_node = rb_entry(node, struct stable_node, node);
			if (stable_node_chain_remove_range(stable_node,
							   start_pfn, end_pfn,
							   root_stable_tree +
							   nid))
				node = rb_first(root_stable_tree + nid);
			else
				node = rb_next(node);
			cond_resched();
		}
	}
	list_for_each_entry_safe(stable_node, next, &migrate_nodes, list) {
		if (stable_node->kpfn >= start_pfn &&
		    stable_node->kpfn < end_pfn)
			remove_node_from_stable_tree(stable_node);
		cond_resched();
	}
}

static int ksm_memory_callback(struct notifier_block *self,
			       unsigned long action, void *arg)
{
	struct memory_notify *mn = arg;

	switch (action) {
	case MEM_GOING_OFFLINE:
		/*
		 * Prevent ksm_do_scan(), unmerge_and_remove_all_rmap_items()
		 * and remove_all_stable_nodes() while memory is going offline:
		 * it is unsafe for them to touch the stable tree at this time.
		 * But unmerge_ksm_pages(), rmap lookups and other entry points
		 * which do not need the ksm_thread_mutex are all safe.
		 */
		mutex_lock(&ksm_thread_mutex);
		ksm_run |= KSM_RUN_OFFLINE;
		mutex_unlock(&ksm_thread_mutex);
		break;

	case MEM_OFFLINE:
		/*
		 * Most of the work is done by page migration; but there might
		 * be a few stable_nodes left over, still pointing to struct
		 * pages which have been offlined: prune those from the tree,
		 * otherwise get_ksm_page() might later try to access a
		 * non-existent struct page.
		 */
		ksm_check_stable_tree(mn->start_pfn,
				      mn->start_pfn + mn->nr_pages);
		/* fallthrough */

	case MEM_CANCEL_OFFLINE:
		mutex_lock(&ksm_thread_mutex);
		ksm_run &= ~KSM_RUN_OFFLINE;
		mutex_unlock(&ksm_thread_mutex);

		smp_mb();	/* wake_up_bit advises this */
		wake_up_bit(&ksm_run, ilog2(KSM_RUN_OFFLINE));
		break;
	}
	return NOTIFY_OK;
}
#else
static void wait_while_offlining(void)
{
}
#endif /* CONFIG_MEMORY_HOTREMOVE */

#ifdef CONFIG_SYSFS
/*
 * This all compiles without CONFIG_SYSFS, but is a waste of space.
 */

#define KSM_ATTR_RO(_name) \
	static struct kobj_attribute _name##_attr = __ATTR_RO(_name)
#define KSM_ATTR(_name) \
	static struct kobj_attribute _name##_attr = \
		__ATTR(_name, 0644, _name##_show, _name##_store)

static ssize_t sleep_millisecs_show(struct kobject *kobj,
				    struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", ksm_thread_sleep_millisecs);
}

static ssize_t sleep_millisecs_store(struct kobject *kobj,
				     struct kobj_attribute *attr,
				     const char *buf, size_t count)
{
	unsigned int msecs;
	int err;

	err = kstrtouint(buf, 10, &msecs);
	if (err)
		return -EINVAL;

	ksm_thread_sleep_millisecs = msecs;
	wake_up_interruptible(&ksm_iter_wait);

	return count;
}
KSM_ATTR(sleep_millisecs);

static ssize_t pages_to_scan_show(struct kobject *kobj,
				  struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", ksm_thread_pages_to_scan);
}

static ssize_t pages_to_scan_store(struct kobject *kobj,
				   struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	unsigned int nr_pages;
	int err;

	err = kstrtouint(buf, 10, &nr_pages);
	if (err)
		return -EINVAL;

	ksm_thread_pages_to_scan = nr_pages;

	return count;
}
KSM_ATTR(pages_to_scan);

static ssize_t run_show(struct kobject *kobj, struct kobj_attribute *attr,
			char *buf)
{
	if (ksm_run & KSM_RUN_ONESHOT)
		return sysfs_emit(buf, "%u\n", KSM_RUN_ONESHOT);
	else
		return sysfs_emit(buf, "%lu\n", ksm_run);
}

static ssize_t run_store(struct kobject *kobj, struct kobj_attribute *attr,
			 const char *buf, size_t count)
{
	unsigned int flags;
	int err;

	err = kstrtouint(buf, 10, &flags);
	if (err)
		return -EINVAL;
	if (flags > KSM_RUN_ONESHOT)
		return -EINVAL;

	/*
	 * KSM_RUN_MERGE sets ksmd running, and 0 stops it running.
	 * KSM_RUN_UNMERGE stops it running and unmerges all rmap_items,
	 * breaking COW to free the pages_shared (but leaves mm_slots
	 * on the list for when ksmd may be set running again).
	 */

	mutex_lock(&ksm_thread_mutex);
	wait_while_offlining();
	if (ksm_run != flags) {
		if (flags == KSM_RUN_ONESHOT)
			ksm_run = KSM_RUN_MERGE | KSM_RUN_ONESHOT;
		else
			ksm_run = flags;
		if (flags & KSM_RUN_UNMERGE) {
			set_current_oom_origin();
			err = unmerge_and_remove_all_rmap_items();
			clear_current_oom_origin();
			if (err) {
				ksm_run = KSM_RUN_STOP;
				count = err;
			}
		}
	}
	mutex_unlock(&ksm_thread_mutex);

	if (ksm_run & KSM_RUN_MERGE) {
		ksm_debug("activate KSM");
		wake_up(&ksm_crawl_wait);
	}

	return count;
}
KSM_ATTR(run);

#ifdef CONFIG_NUMA
static ssize_t merge_across_nodes_show(struct kobject *kobj,
				struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", ksm_merge_across_nodes);
}

static ssize_t merge_across_nodes_store(struct kobject *kobj,
				   struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	int err;
	unsigned long knob;

	err = kstrtoul(buf, 10, &knob);
	if (err)
		return err;
	if (knob > 1)
		return -EINVAL;

	mutex_lock(&ksm_thread_mutex);
	wait_while_offlining();
	if (ksm_merge_across_nodes != knob) {
		if (ksm_pages_shared || remove_all_stable_nodes())
			err = -EBUSY;
		else if (root_stable_tree == one_stable_tree) {
			struct rb_root *buf;
			/*
			 * This is the first time that we switch away from the
			 * default of merging across nodes: must now allocate
			 * a buffer to hold as many roots as may be needed.
			 * Allocate stable and unstable together:
			 * MAXSMP NODES_SHIFT 10 will use 16kB.
			 */
			buf = kcalloc(nr_node_ids + nr_node_ids, sizeof(*buf),
				      GFP_KERNEL);
			/* Let us assume that RB_ROOT is NULL is zero */
			if (!buf)
				err = -ENOMEM;
			else {
				root_stable_tree = buf;
				root_unstable_tree = buf + nr_node_ids;
				/* Stable tree is empty but not the unstable */
				root_unstable_tree[0] = one_unstable_tree[0];
			}
		}
		if (!err) {
			ksm_merge_across_nodes = knob;
			ksm_nr_node_ids = knob ? 1 : nr_node_ids;
		}
	}
	mutex_unlock(&ksm_thread_mutex);

	return err ? err : count;
}
KSM_ATTR(merge_across_nodes);
#endif

static ssize_t use_zero_pages_show(struct kobject *kobj,
				struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", ksm_use_zero_pages);
}
static ssize_t use_zero_pages_store(struct kobject *kobj,
				   struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	int err;
	bool value;

	err = kstrtobool(buf, &value);
	if (err)
		return -EINVAL;

	ksm_use_zero_pages = value;

	return count;
}
KSM_ATTR(use_zero_pages);

static ssize_t max_page_sharing_show(struct kobject *kobj,
				     struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", ksm_max_page_sharing);
}

static ssize_t max_page_sharing_store(struct kobject *kobj,
				      struct kobj_attribute *attr,
				      const char *buf, size_t count)
{
	int err;
	int knob;

	err = kstrtoint(buf, 10, &knob);
	if (err)
		return err;
	/*
	 * When a KSM page is created it is shared by 2 mappings. This
	 * being a signed comparison, it implicitly verifies it's not
	 * negative.
	 */
	if (knob < 2)
		return -EINVAL;

	if (READ_ONCE(ksm_max_page_sharing) == knob)
		return count;

	mutex_lock(&ksm_thread_mutex);
	wait_while_offlining();
	if (ksm_max_page_sharing != knob) {
		if (ksm_pages_shared || remove_all_stable_nodes())
			err = -EBUSY;
		else
			ksm_max_page_sharing = knob;
	}
	mutex_unlock(&ksm_thread_mutex);

	return err ? err : count;
}
KSM_ATTR(max_page_sharing);

static ssize_t pages_shared_show(struct kobject *kobj,
				 struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%lu\n", ksm_pages_shared);
}
KSM_ATTR_RO(pages_shared);

static ssize_t pages_sharing_show(struct kobject *kobj,
				  struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%lu\n", ksm_pages_sharing);
}
KSM_ATTR_RO(pages_sharing);

static ssize_t pages_unshared_show(struct kobject *kobj,
				   struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%lu\n", ksm_pages_unshared);
}
KSM_ATTR_RO(pages_unshared);

static ssize_t pages_volatile_show(struct kobject *kobj,
				   struct kobj_attribute *attr, char *buf)
{
	long ksm_pages_volatile;

	ksm_pages_volatile = ksm_rmap_items - ksm_pages_shared
				- ksm_pages_sharing - ksm_pages_unshared;
	/*
	 * It was not worth any locking to calculate that statistic,
	 * but it might therefore sometimes be negative: conceal that.
	 */
	if (ksm_pages_volatile < 0)
		ksm_pages_volatile = 0;
	return sysfs_emit(buf, "%ld\n", ksm_pages_volatile);
}
KSM_ATTR_RO(pages_volatile);

static ssize_t stable_node_dups_show(struct kobject *kobj,
				     struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%lu\n", ksm_stable_node_dups);
}
KSM_ATTR_RO(stable_node_dups);

static ssize_t stable_node_chains_show(struct kobject *kobj,
				       struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%lu\n", ksm_stable_node_chains);
}
KSM_ATTR_RO(stable_node_chains);

static ssize_t
stable_node_chains_prune_millisecs_show(struct kobject *kobj,
					struct kobj_attribute *attr,
					char *buf)
{
	return sysfs_emit(buf, "%u\n", ksm_stable_node_chains_prune_millisecs);
}

static ssize_t
stable_node_chains_prune_millisecs_store(struct kobject *kobj,
					 struct kobj_attribute *attr,
					 const char *buf, size_t count)
{
	unsigned int msecs;
	int err;

	err = kstrtouint(buf, 10, &msecs);
	if (err)
		return -EINVAL;

	ksm_stable_node_chains_prune_millisecs = msecs;

	return count;
}
KSM_ATTR(stable_node_chains_prune_millisecs);

static ssize_t full_scans_show(struct kobject *kobj,
			       struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%lu\n", ksm_scan.nr_full_scan);
}
KSM_ATTR_RO(full_scans);

static ssize_t scanning_process_show(struct kobject *kobj,
					struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", ksm_nr_added_process);
}
KSM_ATTR_RO(scanning_process);

static ssize_t full_scan_interval_show(struct kobject *kobj,
	struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%lu\n", full_scan_interval);
}

static ssize_t full_scan_interval_store(struct kobject *kbj,
	struct kobj_attribute *attr, const char *buf, size_t count)
{
	unsigned int interval;
	int err;

	err = kstrtouint(buf, 10, &interval);
	if (err)
		return -EINVAL;

	full_scan_interval = interval;
	return count;
}
KSM_ATTR(full_scan_interval);

static ssize_t one_shot_scanning_show(struct kobject *kobj,
					struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%d\n", atomic_read(&ksm_one_shot_scanning));
}

static ssize_t one_shot_scanning_store(struct kobject *kbj,
	struct kobj_attribute *attr, const char *buf, size_t count)
{
	int err, val;

	err = kstrtoint(buf, 10, &val);
	if (err || (val != LKSM_SCAN_PARTIAL && val != LKSM_SCAN_FULL)) {
		ksm_err("wrong value: %d", val);
		return -EINVAL;
	}

	if (!atomic_cmpxchg(&ksm_one_shot_scanning, LKSM_SCAN_NONE, val)) {
		wake_up(&ksm_crawl_wait);
		return count;
	}
	ksm_debug("ksm is still scanning");
	return -EINVAL;
}
KSM_ATTR(one_shot_scanning);

static ssize_t scan_boost_show(struct kobject *kobj,
                   struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", lksm_boosted_pages_to_scan);
}

static ssize_t scan_boost_store(struct kobject *kbj,
   struct kobj_attribute *attr, const char *buf, size_t count)
{
	int err, val;

	err = kstrtoint(buf, 10, &val);
	/* lksm_boosted_pages_to_scan must presence in from 100 to 10000 */
	if (err || val < 100 || val > 10000) {
		ksm_err("wrong value: %d", val);
		return -EINVAL;
	}

	lksm_boosted_pages_to_scan = (unsigned int) val;

	return count;
}
KSM_ATTR(scan_boost);

#ifdef CONFIG_LKSM_FILTER
static ssize_t nr_regions_show(struct kobject *kobj,
				 struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", lksm_nr_regions);
}
KSM_ATTR_RO(nr_regions);

static ssize_t region_share_show(struct kobject *obj,
				struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%s:%d %s:%d %s:%d %s:%d %s:%d\n",
			region_type_str[0], region_share[0], region_type_str[1], region_share[1],
			region_type_str[2], region_share[2], region_type_str[3], region_share[3],
			region_type_str[4], region_share[4]);
}
KSM_ATTR_RO(region_share);
#endif /* CONFIG_LKSM_FILTER */

static struct attribute *ksm_attrs[] = {
	&sleep_millisecs_attr.attr,
	&pages_to_scan_attr.attr,
	&run_attr.attr,
	&pages_shared_attr.attr,
	&pages_sharing_attr.attr,
	&pages_unshared_attr.attr,
	&pages_volatile_attr.attr,
	&full_scans_attr.attr,
#ifdef CONFIG_NUMA
	&merge_across_nodes_attr.attr,
#endif
	&max_page_sharing_attr.attr,
	&stable_node_chains_attr.attr,
	&stable_node_dups_attr.attr,
	&stable_node_chains_prune_millisecs_attr.attr,
	&use_zero_pages_attr.attr,
	&scanning_process_attr.attr,
	&full_scan_interval_attr.attr,
	&one_shot_scanning_attr.attr,
	&scan_boost_attr.attr,
#ifdef CONFIG_LKSM_FILTER
	&nr_regions_attr.attr,
	&region_share_attr.attr,
#endif
	NULL,
};

static const struct attribute_group ksm_attr_group = {
	.attrs = ksm_attrs,
	.name = "ksm",
};
#endif /* CONFIG_SYSFS */

#ifdef CONFIG_LKSM_FILTER
static inline void init_lksm_region
(struct lksm_region *region, unsigned long ino, int type, unsigned long len)
{
	region->ino = ino;
	region->type = type;
	region->len = len;
}

/* if region is newly allocated, the function returns true. */
static void lksm_insert_region
(struct lksm_region **region, unsigned long ino,
struct vm_area_struct *vma, int type)
{
	int need_hash_add = 0;
	unsigned long len, size;
	struct lksm_region *next = NULL;
	unsigned long flags;

	size = lksm_region_size(vma->vm_start, vma->vm_end);
	len = (size > BITS_PER_LONG) ? lksm_bitmap_size(size) : SINGLE_FILTER_LEN;

	if (!(*region)) {
		*region = kzalloc(sizeof(struct lksm_region), GFP_KERNEL);
		if (!*region) {
			ksm_err("region allocation failed");
			return;
		}
		init_lksm_region(*region, ino, LKSM_REGION_FILE1, len);
		(*region)->scan_round = ksm_crawl_round;
		atomic_set(&(*region)->refcount, 0);
		lksm_nr_regions++;
		need_hash_add = 1;
	}

	if (!(*region)->next && type == LKSM_REGION_FILE2) {
		next = kzalloc(sizeof(struct lksm_region), GFP_KERNEL);
		if (!next) {
			if (need_hash_add)
				kfree(*region);
			*region = NULL;
			ksm_err("region allocation failed");
			return;
		}
		init_lksm_region(next, ino, LKSM_REGION_FILE2, len);
		atomic_set(&next->refcount, 0);
		next->scan_round = ksm_crawl_round;
		lksm_nr_regions++;
	}

	if (need_hash_add || next) {
		spin_lock_irqsave(&lksm_region_lock, flags);
		if (need_hash_add)
			hash_add(lksm_region_hash, &(*region)->hnode, ino);
		if (next) {
			(*region)->next = next;
			next->prev = *region;
		}
		spin_unlock_irqrestore(&lksm_region_lock, flags);
	}
}

static inline struct lksm_region *lksm_hash_find_region(unsigned long ino)
{
	struct lksm_region *region;

	hash_for_each_possible(lksm_region_hash, region, hnode, ino)
		if (region->ino == ino)
			return region;
	return NULL;
}

static void lksm_register_file_anon_region
(struct mm_slot *slot, struct vm_area_struct *vma)
{
	struct lksm_region *region;
	struct file *file = NULL;
	struct inode *inode;
	unsigned long flags;
	int type;

	if (vma->vm_file) {
		file = vma->vm_file;
		type = LKSM_REGION_FILE1;
	} else if (vma->vm_prev) {
		/* LKSM should deal with .NET libraries */
		struct vm_area_struct *prev = vma->vm_prev;
		if (prev->vm_flags & VM_MERGEABLE && prev->vm_file) {
			/* Linux standard map structure */
			file = prev->vm_file;
			type = LKSM_REGION_FILE2;
		} else {
			/* DLL map structure */
			int i = 0;
			bool find = false;
			while (i <= LKSM_REGION_ITER_MAX && prev) {
				if (file == NULL)
					file = prev->vm_file;
				else if (prev->vm_file && file != prev->vm_file)
					break;

				if (prev->vm_flags & VM_MERGEABLE && file) {
					find = true;
					break;
				}
				prev = prev->vm_prev;
				i++;
			}
			if (find)
				type = LKSM_REGION_FILE2;
			else
				file = NULL;
		}
	}

	if (file) {
		inode = file_inode(file);
		BUG_ON(!inode);

		spin_lock_irqsave(&lksm_region_lock, flags);
		region = lksm_hash_find_region(inode->i_ino);
		spin_unlock_irqrestore(&lksm_region_lock, flags);

		lksm_insert_region(&region, inode->i_ino, vma, type);
		if (region) {
			if (type == LKSM_REGION_FILE1)
				lksm_region_ref_append(slot, region);
			else
				lksm_region_ref_append(slot, region->next);
		}
	}
}

static struct lksm_region *lksm_find_region(struct vm_area_struct *vma)
{
	struct lksm_region *region = NULL;
	struct file *file = NULL;
	struct inode *inode;
	unsigned long ino = 0, flags;
	int type;

	if (is_heap(vma))
		return &heap_region;
	else if (is_stack(vma))
		return NULL;
	else if (!vma->anon_vma)
		return NULL;
	else if (is_exec(vma))
		return NULL;

	if (vma->vm_file) {
		/* check thread stack */
		file = vma->vm_file;
		type = LKSM_REGION_FILE1;
	} else if (vma->vm_prev) {
		struct vm_area_struct *prev = vma->vm_prev;
		if (prev->vm_flags & VM_MERGEABLE && prev->vm_file) {
			/* Linux standard map structure */
			file = prev->vm_file;
			type = LKSM_REGION_FILE2;
		} else {
			/* DLL map structure */
			int i = 0;
			bool find = false;
			while (i <= LKSM_REGION_ITER_MAX && prev) {
				if (file == NULL)
					file = prev->vm_file;
				else if (prev->vm_file && file != prev->vm_file)
					break;

				if (prev->vm_flags & VM_MERGEABLE && file) {
					find = true;
					break;
				}
				prev = prev->vm_prev;
				i++;
			}
			if (find)
				type = LKSM_REGION_FILE2;
			else
				file = NULL;
		}
	}

	if (file) {
		inode = file_inode(file);
		BUG_ON(!inode);
		ino = inode->i_ino;

		if (ksm_scan.region && ksm_scan.region->ino == ino) {
			if (ksm_scan.region->type == type)
				return ksm_scan.region;
			else if (ksm_scan.region->type == LKSM_REGION_FILE1)
				region = ksm_scan.region;
		} else {
			spin_lock_irqsave(&lksm_region_lock, flags);
			region = lksm_hash_find_region(ino);
			spin_unlock_irqrestore(&lksm_region_lock, flags);
		}
	}

	if (region && type == LKSM_REGION_FILE2) {
		if (!region->next) {
			lksm_insert_region(&region, ino, vma, type);
			BUG_ON(!region->next);
		}
		return region->next;
	}
	return region;
}
#endif /* CONFIG_LKSM_FILTER */

static inline int __lksm_remove_candidate(struct task_struct *task)
{
	int ret = LKSM_TASK_SLOT_NONE;
	struct task_slot *slot = get_task_slot(task);

	if (slot) {
		list_del(&slot->list);
		hash_del(&slot->hlist);
		free_task_slot(slot);
		ret = LKSM_TASK_SLOT_REMOVED;
	}
	return ret;
}

/* called by ksm_exit */
void lksm_remove_candidate(struct mm_struct *mm)
{
	int ret;

	if (!mm->owner) {
		struct mm_slot *mm_slot;

		spin_lock(&ksm_mmlist_lock);
		mm_slot = get_mm_slot(mm);
		if (mm_slot && mm_slot != ksm_scan.mm_slot) {
			list_move(&mm_slot->mm_list, &ksm_scan.remove_mm_list);
			if (lksm_test_mm_state(mm_slot, KSM_MM_FROZEN))
				atomic_dec(&ksm_scan.nr_frozen);
			else if (!lksm_test_mm_state(mm_slot, KSM_MM_SCANNED))
				atomic_dec(&ksm_scan.nr_scannable);
		}
		spin_unlock(&ksm_mmlist_lock);
		return;
	}

	spin_lock(&frozen_task_lock);
	ret = __lksm_remove_candidate(mm->owner);
	spin_unlock(&frozen_task_lock);
	if (ret == LKSM_TASK_SLOT_REMOVED)
		put_task_struct(mm->owner);
}

static int lksm_task_frozen(struct task_struct *task)
{
	int need_wakeup = 0;
	struct mm_struct *mm = task->mm;
	struct mm_slot *mm_slot;
	struct task_slot *task_slot;

	if (mm && test_bit(MMF_VM_MERGEABLE, &mm->flags)) {
		/* a mergeable task becoming frozen */
		spin_lock(&ksm_mmlist_lock);
		mm_slot = get_mm_slot(mm);
		BUG_ON(!mm_slot);

		if (mm_slot != ksm_scan.mm_slot
				&& lksm_test_mm_state(mm_slot, KSM_MM_LISTED)) {
			if (list_empty(&mm_slot->scan_list))
				list_add_tail(&mm_slot->scan_list, &ksm_scan_head.scan_list);
			if (!lksm_test_mm_state(mm_slot, KSM_MM_SCANNED))
				atomic_dec(&ksm_scan.nr_scannable);
			lksm_clear_mm_state(mm_slot, KSM_MM_LISTED);
			lksm_set_mm_state(mm_slot, KSM_MM_FROZEN);
			atomic_inc(&ksm_scan.nr_frozen);

			need_wakeup = (ksm_run == KSM_RUN_MERGE);
			ksm_debug("lksm_task_frozen called for task(%s): %p (nr_frozen: %d)",
					task->comm, task, atomic_read(&ksm_scan.nr_frozen));
		}
		spin_unlock(&ksm_mmlist_lock);
	} else {
		task_slot = alloc_task_slot();
		if (!task_slot) {
			ksm_err("[ksm_tizen] Cannot allocate memory for task_slot\n");
			return -ENOMEM;
		}

		task_slot->task = task;
		task_slot->frozen = KSM_TASK_FROZEN;
		task_slot->inserted = jiffies;

		get_task_struct(task);

		spin_lock(&frozen_task_lock);
		list_add(&task_slot->list, &frozen_task_list);
		insert_to_task_slots_hash(task_slot);
		spin_unlock(&frozen_task_lock);

		need_wakeup = (ksm_run == KSM_RUN_MERGE);
		ksm_debug("task-%d(%s) is added to frozen task list",
				task_pid_nr(task), task->comm);
	}

	if (need_wakeup && atomic_read(&crawl_state) == KSM_CRAWL_SLEEP)
		wake_up(&ksm_crawl_wait);

	return 0;
}

static int lksm_task_thawed(struct task_struct *task)
{
	struct mm_struct *mm = task->mm;
	struct mm_slot *mm_slot;
	struct task_slot *task_slot;

	if (mm && test_bit(MMF_VM_MERGEABLE, &mm->flags)) {
		/* a frozen task becoming thawed */
		spin_lock(&ksm_mmlist_lock);
		mm_slot = get_mm_slot(mm);
		BUG_ON(!mm_slot);

		if (lksm_test_mm_state(mm_slot, KSM_MM_FROZEN)
				&& ksm_scan.mm_slot != mm_slot) {
			if (!lksm_test_mm_state(mm_slot, KSM_MM_SCANNED))
				atomic_inc(&ksm_scan.nr_scannable);
			else
				list_del_init(&mm_slot->scan_list);
			lksm_clear_mm_state(mm_slot, KSM_MM_FROZEN);
			lksm_set_mm_state(mm_slot, KSM_MM_LISTED);
			atomic_dec(&ksm_scan.nr_frozen);
			ksm_debug("nr_frozen: %d nr_scannable: %d",
					atomic_read(&ksm_scan.nr_frozen),
					atomic_read(&ksm_scan.nr_scannable));
		}
		spin_unlock(&ksm_mmlist_lock);
	} else {
		/* just remove task slot, it will be cared by full_scan */
		spin_lock(&frozen_task_lock);
		task_slot = get_task_slot(task);
		if (task_slot) {
			list_del(&task_slot->list);
			hash_del(&task_slot->hlist);
		}
		spin_unlock(&frozen_task_lock);
		if (task_slot) {
			free_task_slot(task_slot);
			put_task_struct(task);
			ksm_debug("task-%d(%s) is removed from frozen task list",
				task_pid_nr(task), task->comm);
		}
	}

	return 0;
}

/*
 * lksm_hint: a hook for construct candidate list
 * this function cannot sleep
 */
int lksm_hint(struct task_struct *task, int frozen)
{
	/*
	 * If lksm_hint is called by ksm_fork, the task yet has its own
	 * mm_struct because it does not completes mm_struct initialization.
	 * Thus, we skip this check and put the task into candidate list.
	 */
	if (frozen == KSM_TASK_FROZEN)
		return lksm_task_frozen(task);
	else if (frozen == KSM_TASK_THAWED)
		return lksm_task_thawed(task);
	else
		return 0;
}

static void __init lksm_init(void)
{
	ksm_crawld = kthread_create(lksm_crawl_thread, NULL, "ksm_crawld");

	if (ksm_crawld == NULL) {
		printk(KERN_ALERT "fail to create ksm crawler daemon\n");
		return;
	}

	atomic_set(&ksm_scan.nr_frozen, 0);
	atomic_set(&ksm_scan.nr_scannable, 0);
	atomic_set(&ksm_state, 0);
	INIT_LIST_HEAD(&ksm_scan.remove_mm_list);

	crawler_sleep = msecs_to_jiffies(1000);
#ifdef CONFIG_LKSM_FILTER
	init_lksm_region(&heap_region, 0, LKSM_REGION_HEAP, 0);
	heap_region.merge_cnt = 0;
	heap_region.filter_cnt = 0;
	heap_region.filter = NULL;

	init_lksm_region(&unknown_region, 0, LKSM_REGION_UNKNOWN, 0);
	unknown_region.merge_cnt = 0;
	unknown_region.filter_cnt = 0;
	unknown_region.filter = NULL;

	spin_lock_init(&lksm_region_lock);
#endif /* CONFIG_LKSM_FILTER */
	wake_up_process(ksm_crawld);
}

static int __init ksm_init(void)
{
	struct task_struct *ksm_thread;
	int err;

	/* The correct value depends on page size and endianness */
	zero_checksum = calc_checksum(ZERO_PAGE(0));
	/* Default to false for backwards compatibility */
	ksm_use_zero_pages = false;

	err = ksm_slab_init();
	if (err)
		goto out;

	ksm_thread = kthread_run(lksm_scan_thread, NULL, "ksmd");
	if (IS_ERR(ksm_thread)) {
		pr_err("ksm: creating kthread failed\n");
		err = PTR_ERR(ksm_thread);
		goto out_free;
	}

#ifdef CONFIG_SYSFS
	err = sysfs_create_group(mm_kobj, &ksm_attr_group);
	if (err) {
		pr_err("ksm: register sysfs failed\n");
		kthread_stop(ksm_thread);
		goto out_free;
	}
#else
	ksm_run = KSM_RUN_MERGE;	/* no way for user to start it */

#endif /* CONFIG_SYSFS */
	lksm_init();
#ifdef CONFIG_MEMORY_HOTREMOVE
	/* There is no significance to this priority 100 */
	hotplug_memory_notifier(ksm_memory_callback, 100);
#endif
	return 0;

out_free:
	ksm_slab_free();
out:
	return err;
}
subsys_initcall(ksm_init);
