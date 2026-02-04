"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.current_bundle = defaultdict(list)
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def flush_bundle(self):
        if self.current_bundle:
            self.instrs.append(dict(self.current_bundle))
            self.current_bundle = defaultdict(list)

    def add(self, engine, slot):
        if len(self.current_bundle[engine]) >= SLOT_LIMITS[engine]:
            self.flush_bundle()
        self.current_bundle[engine].append(slot)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # We've updated add to handle bundling, so this is now just for legacy/manual use
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash_vec(self, v_val, v_tmp1, v_tmp2, round, i_start, hash_const_vecs):
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            bc1, bc3 = hash_const_vecs[hi]
            self.add("valu", (op1, v_tmp1, v_val, bc1))
            self.add("valu", (op3, v_tmp2, v_val, bc3))
            self.flush_bundle()
            self.add("valu", (op2, v_val, v_tmp1, v_tmp2))
            self.flush_bundle()
            # Debug is ignored, so no need to flush after it unless it's for something else
            for vi in range(VLEN):
                self.add("debug", ("compare", v_val + vi, (round, i_start + vi, "hash_stage", hi)))

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Hyper-optimized 4-block unrolled VLIW implementation.
        """
        # --- Metadata & Constants ---
        metadata_addrs = ["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"]
        meta = {name: self.alloc_scratch(name) for name in metadata_addrs}
        curr_ptr = self.alloc_scratch("init_ptr")
        for i, name in enumerate(metadata_addrs):
            self.add("load", ("const", curr_ptr, i))
            self.flush_bundle()
            self.add("load", ("load", meta[name], curr_ptr))
        
        num_blocks = 4
        vlen_total_const = self.scratch_const(VLEN * num_blocks)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        zero_const = self.scratch_const(0)
        
        # Broadcasts
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        self.add("valu", ("vbroadcast", one_vec, one_const))
        self.add("valu", ("vbroadcast", two_vec, two_const))
        self.add("valu", ("vbroadcast", zero_vec, zero_const))
        self.add("valu", ("vbroadcast", n_nodes_vec, meta["n_nodes"]))
        
        hash_const_vecs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.scratch_const(val1, f"h_{hi}_v1")
            c3 = self.scratch_const(val3, f"h_{hi}_v3")
            self.flush_bundle()
            bc1 = self.alloc_scratch(f"h_{hi}_c1", VLEN)
            bc3 = self.alloc_scratch(f"h_{hi}_c3", VLEN)
            self.add("valu", ("vbroadcast", bc1, c1))
            self.add("valu", ("vbroadcast", bc3, c3))
            hash_const_vecs.append((bc1, bc3))
        
        # Scratch Vectors
        block_v_idx = [self.alloc_scratch(f"v_idx_{b}", VLEN) for b in range(num_blocks)]
        block_v_val = [self.alloc_scratch(f"v_val_{b}", VLEN) for b in range(num_blocks)]
        block_v_node_val = [self.alloc_scratch(f"v_node_val_{b}", VLEN) for b in range(num_blocks)]
        block_v_tmp1 = [self.alloc_scratch(f"v_tmp1_{b}", VLEN) for b in range(num_blocks)]
        block_v_tmp2 = [self.alloc_scratch(f"v_tmp2_{b}", VLEN) for b in range(num_blocks)]
        block_addr_vec = [self.alloc_scratch(f"addr_vec_{b}", VLEN) for b in range(num_blocks)]
        
        # Persistent Pointers
        inp_idx_pts = [self.alloc_scratch(f"ptr_idx_{b}") for b in range(num_blocks)]
        inp_val_pts = [self.alloc_scratch(f"ptr_val_{b}") for b in range(num_blocks)]

        self.flush_bundle()
        self.add("flow", ("pause",))
        self.flush_bundle()

        # Initial pointer offsets (only once)
        for b in range(num_blocks):
            off = self.scratch_const(b * VLEN)
            self.flush_bundle()
            self.add("alu", ("+", inp_idx_pts[b], meta["inp_indices_p"], off))
            self.add("alu", ("+", inp_val_pts[b], meta["inp_values_p"], off))
        self.flush_bundle()

        for round in range(rounds):
            # Reset pointers to round start
            for b in range(num_blocks):
                off = self.scratch_const(b * VLEN)
                self.add("alu", ("+", inp_idx_pts[b], meta["inp_indices_p"], off))
                self.add("alu", ("+", inp_val_pts[b], meta["inp_values_p"], off))
            self.flush_bundle()
            
            for batch_start in range(0, batch_size, VLEN * num_blocks):
                # 1. Load Data
                for b in range(num_blocks):
                    self.add("load", ("vload", block_v_idx[b], inp_idx_pts[b]))
                    self.add("load", ("vload", block_v_val[b], inp_val_pts[b]))
                self.flush_bundle()
                
                # 2. Gather Node Addresses
                for b in range(num_blocks):
                    for vi in range(VLEN):
                        self.add("alu", ("+", block_addr_vec[b] + vi, meta["forest_values_p"], block_v_idx[b] + vi))
                self.flush_bundle()
                
                # 3. Gather Node Loads
                for b in range(num_blocks):
                    for vi in range(0, VLEN, 2):
                        self.add("load", ("load", block_v_node_val[b] + vi, block_addr_vec[b] + vi))
                        self.add("load", ("load", block_v_node_val[b] + vi + 1, block_addr_vec[b] + vi + 1))
                        self.flush_bundle()
                
                # 4. Processing Core - XOR
                for b in range(num_blocks):
                    self.add("valu", ("^", block_v_val[b], block_v_val[b], block_v_node_val[b]))
                self.flush_bundle()
                
                # Hash Stages
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    bc1, bc3 = hash_const_vecs[hi]
                    for b in range(num_blocks):
                        self.add("valu", (op1, block_v_tmp1[b], block_v_val[b], bc1))
                        self.add("valu", (op3, block_v_tmp2[b], block_v_val[b], bc3))
                    self.flush_bundle()
                    for b in range(num_blocks):
                        self.add("valu", (op2, block_v_val[b], block_v_tmp1[b], block_v_tmp2[b]))
                    self.flush_bundle()
                
                # 5. Index Update
                for b in range(num_blocks):
                    self.add("valu", ("&", block_v_tmp1[b], block_v_val[b], one_vec))
                self.flush_bundle()
                for b in range(num_blocks):
                    self.add("valu", ("+", block_v_tmp1[b], block_v_tmp1[b], one_vec))
                self.flush_bundle()
                for b in range(num_blocks):
                    self.add("valu", ("multiply_add", block_v_idx[b], block_v_idx[b], two_vec, block_v_tmp1[b]))
                self.flush_bundle()
                
                # 6. Wrap & Store
                for b in range(num_blocks):
                    self.add("valu", ("<", block_v_tmp1[b], block_v_idx[b], n_nodes_vec))
                self.flush_bundle()
                for b in range(num_blocks):
                    self.add("flow", ("vselect", block_v_idx[b], block_v_tmp1[b], block_v_idx[b], zero_vec))
                self.flush_bundle()
                
                for b in range(num_blocks):
                    self.add("store", ("vstore", inp_idx_pts[b], block_v_idx[b]))
                    self.add("store", ("vstore", inp_val_pts[b], block_v_val[b]))
                    self.add("alu", ("+", inp_idx_pts[b], inp_idx_pts[b], vlen_total_const))
                    self.add("alu", ("+", inp_val_pts[b], inp_val_pts[b], vlen_total_const))
                self.flush_bundle()

        self.add("flow", ("pause",))
        self.flush_bundle()

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        if i == 0: continue # Skip if you want, but machine.run() must be called
        inp_values_p = ref_mem[6]
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256, prints=False)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
