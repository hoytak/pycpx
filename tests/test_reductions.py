from common import *

class Test(unittest.TestCase):

    def test01_sum_01(self):
        m = CPlexModel()
        x = m.new( (2, 2), name = "x")

        A = arange(1, 5).reshape( (2, 2) )

        m.constrain( x.sum(axis = 0) <= 1)
        m.constrain( x >= 0)


        m.maximize( (A * x.A).sum())

        self.assertEqual(x[0, 0], 0)
        self.assertEqual(x[0, 1], 1)
        self.assertEqual(x[1, 0], 0)
        self.assertEqual(x[1, 1], 1)


    def test01_sum_02(self):
        m = CPlexModel()
        x = m.new( (2, 2), name = "x")

        A = arange(1, 5).reshape( (2, 2) )

        m.constrain( x.T.sum(axis = 1) <= ar([1,2]))
        m.constrain( x >= 0)


        m.maximize( (A * x.A).sum())

        self.assertEqual(x[0, 0], 0)
        self.assertEqual(x[0, 1], 1)
        self.assertEqual(x[1, 0], 0)
        self.assertEqual(x[1, 1], 2)


if __name__ == '__main__':
    unittest.main()

