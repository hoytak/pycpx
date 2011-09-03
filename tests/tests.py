#!/usr/bin/env python


import unittest, sys

if __name__ == '__main__':
    dtl = unittest.defaultTestLoader

    import test_basic
    import test_slicing
    import test_reductions
    
    ts = unittest.TestSuite([
        dtl.loadTestsFromModule(test_basic),
        dtl.loadTestsFromModule(test_slicing),
        dtl.loadTestsFromModule(test_reductions),
        ])

    if '--verbose' in sys.argv:
        unittest.TextTestRunner(verbosity=2).run(ts)
    else:
        unittest.TextTestRunner().run(ts)
