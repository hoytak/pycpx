from common import *

class Test(unittest.TestCase):

    def test01_varslice(self):
        m = CPlexModel()
        x = m.new(4, name = "x")
        m.constrain(x[0:2] <= 1)
        m.constrain(x[2:4] <= 2)
        m.maximize(x[0] + x[1] + x[2] + x[3])
        
        self.assert_( (m[x[:2]] == 1).all())
        v = m[x[2:4]]
        self.assert_( (v == 2).all(), "m[x[2:4]] = %s, not 2" % (str(v)))

    def test02_varblock_01(self):
        m = CPlexModel()

        x = m.new( (4, 4), name = "x")

        m.constrain( (x - eye(4)) <= ones( (4,4) ) )

        m.maximize(ones( (1,4) )* x * ones( (4, 1) ) )

        X = m[x]

        self.assert_(X.shape == (4,4))
        self.assert_( ((X - eye(4)) == 1).all())
    

if __name__ == '__main__':
    unittest.main()

