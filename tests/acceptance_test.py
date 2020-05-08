# -*- coding: future_fstrings -*-
import os
import unittest
import subprocess
from nose.tools import timed
import shutil
import time

class mk_reduce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        unittest.TestCase.setUpClass()
        cls.__test_data_dir = os.environ["TEST_DATA_DIR"]
        cls.__test_output_dir = os.environ["TEST_OUTPUT_DIR"]
        if not os.path.isdir(cls.__test_data_dir) or \
           not os.path.isdir(cls.__test_output_dir):
           raise RuntimeError("Expect TEST_DATA_DIR and TEST_OUTPUT_DIR to exist")
        os.mkdir(os.path.join(cls.__test_output_dir, "msdir"))
        args = "tar xvzf {} -C {}".format(os.path.join(cls.__test_data_dir, "vermeerkat_test.ms.tar.gz"),
                                          os.path.join(cls.__test_output_dir, "msdir"))
        subprocess.check_call(args, 
            shell=True)
        os.chdir(cls.__test_output_dir)

    @classmethod
    def tearDownClass(cls):
        unittest.TestCase.tearDownClass()
        shutil.rmtree(os.path.join("msdir"))

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    def setUp(self):
        unittest.TestCase.setUp(self)

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
                "--containerization", "docker",
                "--gc", "J1726-5529", 
                "--bp", "J1938-6341", 
                "--altcal", "J1331+3030", 
                "--tar", "J1638.2-6420",
                "--ref_ant", "m013", 
                "--time_sol_interval", "inf",
                "--dont_prompt",
                "vermeerkat_test"
                ]
        self.callvermeerkat(args)
        args = ["vermeerkat", 
                "selfcal", 
                "J1638.2-6420.vermeerkat_test.1gc",
                "--ref_ant", "m013",
                "--tar", "J1638.2-6420",
                "--containerization", "docker", 
                "--ncubical_workers", "20",
                "--recipe", "dp(35,16s), dd(45,4.0,32,64,CORRECTED_DATA,DE_DATA)"]
        self.callvermeerkat(args)