# -*- coding: future_fstrings -*-
import os
import unittest
import subprocess
from nose.tools import timed
import shutil
import time

test_container_tech="singularity"

class mk_reduce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        unittest.TestCase.setUpClass()
        cls.__test_data_dir = os.environ["TEST_DATA_DIR"]
        cls.__test_output_dir = os.environ["TEST_OUTPUT_DIR"]
        if not os.path.isdir(cls.__test_data_dir) or \
           not os.path.isdir(cls.__test_output_dir):
           raise RuntimeError("Expect TEST_DATA_DIR and TEST_OUTPUT_DIR to exist")
        os.chdir(cls.__test_output_dir)

    @classmethod
    def tearDownClass(cls):
        unittest.TestCase.tearDownClass()
        if os.path.exists("msdir"):
            shutil.rmtree(os.path.join("msdir"))

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        if os.path.exists("msdir"):
            shutil.rmtree(os.path.join("msdir"))

    def setUp(self):
        unittest.TestCase.setUp(self)

        if os.path.exists("msdir"):
            shutil.rmtree(os.path.join("msdir"))

        os.mkdir(os.path.join("msdir"))
        args = "tar xvzf {} -C {}".format(os.path.join(self.__test_data_dir, "vermeerkat_test.ms.tar.gz"),
                                          os.path.join("msdir"))
        subprocess.check_call(args, 
            shell=True)

    def callvermeerkat(self, args):
        p = subprocess.Popen(args, 
                env=os.environ.copy())
        #not necessary
        #          stdout=stdout_file, 
        #         stderr=stderr_file)
        x = 100000000
        delay = 1.0
        timeout = int(x / delay)
        while p.poll() is None and timeout > 0:
            time.sleep(delay)
            timeout -= delay
        #timeout reached, kill process if it is still rolling
        ret = p.poll()
        if ret is None:
            p.kill()
            ret = 99

        if ret == 99:
            raise RuntimeError("Test timeout reached. Killed process.")
        elif ret != 0:
            raise RuntimeError("VermeerKAT exited with non-zero return code %d" % ret)

    def test_fieldlistr(self):
        args = ["vermeerkat",
                "fieldlist", 
                "vermeerkat_test"
                ]
        self.callvermeerkat(args)
    
    def test_antlistr(self):
        args = ["vermeerkat",
                "antlist", 
                "vermeerkat_test"
                ]
        self.callvermeerkat(args)

    def test_transfer(self):
        args = ["vermeerkat",
                "transfer", 
                "--noncurses",
                "--containerization", test_container_tech,
                "--gc", "J1120-2508", 
                "--bp", "J1939-6342", 
                "--altcal", "J1331+3030", 
                "--tar", "A1300",
                "--ref_ant", "m060", 
                "--time_sol_interval", "inf",
                "--dont_prompt",
                "--skip_final_1GC",
                "vermeerkat_test"
                ]
        self.callvermeerkat(args)
        args = ["vermeerkat", 
                "selfcal", 
                "A1300.vermeerkat_test.1gc",
                "--noncurses",
                "--ref_ant", "m060",
                "--ncc", "1500",
                "--tar", "A1300",
                "--containerization", test_container_tech, 
                "--ncubical_workers", "4",
                "--nfacet", "10",
                "--dont_prompt",
                "--recipe", "dp(35,16s),dd(45,9.0,8,32,CORRECTED_DATA,DE_DATA)"]
        self.callvermeerkat(args)

    def test_xtransfer(self):
        args = ["vermeerkat",
                "transfer", 
                "--noncurses",
                "--containerization", test_container_tech,
                "--gc", "J1120-2508", 
                "--bp", "J1939-6342", 
                "--altcal", "J1331+3030", 
                "--tar", "A1300",
                "--ref_ant", "m060", 
                "--time_sol_interval", "inf",
                "--dont_prompt",
                "--skip_final_1GC",
                "--skip_final_flagging",
                "vermeerkat_test"
                ]
        self.callvermeerkat(args)
        args = ["vermeerkat", "poltransfer",
                "--noncurses",
                "--dont_prompt",
                "--containerization", test_container_tech,
                "--bp", "J1939-6342",
                "--polcal", "J1331+3030", 
                "--refant", "m060", 
                "vermeerkat_test",
                "--transfer_to_existing",
                "A1300.vermeerkat_test.1gc"]
        self.callvermeerkat(args)

