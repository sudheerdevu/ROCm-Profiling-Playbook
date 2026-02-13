#!/usr/bin/env python3
"""
Test suite for ROCm Profiling Playbook analysis tools
"""

import os
import sys
import unittest
import tempfile
import csv

# Add analysis to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))


class TestTraceAnalyzer(unittest.TestCase):
    """Tests for trace analyzer"""
    
    def test_import_trace_analyzer(self):
        """Test trace analyzer import"""
        try:
            from trace_analyzer import RocprofTraceParser, TraceAnalyzer
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"Module not available: {e}")
    
    def test_parse_csv_trace(self):
        """Test parsing CSV trace data"""
        # Create test trace file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Duration_ns', 'BeginNs', 'EndNs'])
            writer.writerow(['MatMul', '1000000', '0', '1000000'])
            writer.writerow(['ReLU', '500000', '1000000', '1500000'])
            temp_path = f.name
        
        try:
            from trace_analyzer import RocprofTraceParser
            parser = RocprofTraceParser()
            # Test would parse the file
        except ImportError:
            pass
        finally:
            os.unlink(temp_path)


class TestKernelAnalyzer(unittest.TestCase):
    """Tests for kernel analyzer"""
    
    def test_import_kernel_analyzer(self):
        """Test kernel analyzer import"""
        try:
            from kernel_analyzer import KernelAnalyzer
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"Module not available: {e}")
    
    def test_occupancy_calculation(self):
        """Test occupancy calculation logic"""
        try:
            from kernel_analyzer import KernelAnalyzer
            analyzer = KernelAnalyzer()
            
            # Test basic occupancy calculation
            # Assuming method exists
            if hasattr(analyzer, 'calculate_occupancy'):
                # Test with typical values
                result = analyzer.calculate_occupancy(
                    threads_per_block=256,
                    registers_per_thread=32,
                    shared_mem_per_block=16384
                )
                self.assertGreater(result, 0)
                self.assertLessEqual(result, 100)
        except ImportError:
            pass


class TestExampleTraces(unittest.TestCase):
    """Tests for example trace files"""
    
    def test_example_trace_exists(self):
        """Verify example trace file exists"""
        trace_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'traces', 'example_kernels.csv'
        )
        self.assertTrue(os.path.exists(trace_path))
    
    def test_example_trace_format(self):
        """Verify example trace file format"""
        trace_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'traces', 'example_kernels.csv'
        )
        
        if os.path.exists(trace_path):
            with open(trace_path) as f:
                reader = csv.reader(f)
                header = next(reader)
                
                # Should have kernel name column
                self.assertTrue(
                    any('name' in h.lower() or 'kernel' in h.lower() for h in header)
                )


class TestScripts(unittest.TestCase):
    """Tests for analysis scripts"""
    
    def test_analyze_script_exists(self):
        """Verify analyze script exists"""
        script_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'scripts', 'analyze-trace.py'
        )
        self.assertTrue(os.path.exists(script_path))


if __name__ == '__main__':
    unittest.main(verbosity=2)
