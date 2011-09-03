from common import *
import tempfile, os

class TestBasic(unittest.TestCase):

    def test01_scalar_01(self):
        m = CPlexModel()
        x = m.new()
        m.constrain(x <= 1)
        m.maximize(x)

        self.assert_(m[x] == 1)

    def test01_scalar_02(self):
        m = CPlexModel()
        x = m.new()
        m.constrain(2*x <= 2)
        m.maximize(x)

        self.assert_(m[x] == 1)

    def test02_vect_01(self):
        m = CPlexModel()
        x = m.new(2)
        m.constrain(x <= 1)
        m.maximize(x[0] + x[1])

        xv = getWithDimensionCheck(m, x, 2)
        
        self.assert_(xv[0] == 1)
        self.assert_(xv[1] == 1)

    def test02_vect_02(self):

        m = CPlexModel()
        x = m.new(2, name = 'x')
        m.constrain(eye(2) * x <= 1)
        m.maximize(x[0] + x[1])

        xv = getWithDimensionCheck(m, x, 2)
        
        self.assertEqual(xv[0], 1)
        self.assertEqual(xv[1], 1)

    def test02_vect_03(self):

        m = CPlexModel()
        x = m.new(2, name = 'x')
        m.constrain(eye(2) * x <= ar([3,4]))
        m.maximize(x[0] + x[1])

        xv = getWithDimensionCheck(m, x, 2)
        
        self.assertEqual(xv[0], 3)
        self.assertEqual(xv[1], 4)

    def test03_shapes_01(self):
        m = CPlexModel()
        x = m.new(10)
        self.assert_(x.shape == (10, 1) )

    def test03_shapes_02(self):
        m = CPlexModel()
        x = m.new(10)
        self.assert_(x.T.shape == (1, 10) )

    def test04_bounds_01(self):
        m = CPlexModel()
        x = m.new(name='x', lb = -1, ub = 1)

        self.assertEqual(m.minimize(x), -1)
        self.assertEqual(m.maximize(x), 1)

    def test04_bounds_02(self):
        m = CPlexModel()
        x = m.new(3, name='x', lb = -1, ub = 1)

        self.assertEqual(m.minimize(x.sum()), -3)
        self.assertEqual(m.maximize(x.sum()), 3)

    def test04_bounds_02(self):
        m = CPlexModel()

        lb = ar([-1,-2,-3])
        ub = ar([1,3,2])
        x = m.new(3, name='x', lb = lb, ub = ub)

        self.assertEqual(m.minimize(x.sum()), -6)
        self.assert_( (m[x] == lb).all() )
        
        self.assertEqual(m.maximize(x.sum()), 6)
        self.assert_( (m[x] == ub).all() )

    def test04_bounds_02(self):
        m = CPlexModel()

        lb = [-1,None,-3]
        ub = [-1,None,-3]
        x = m.new(3, name='x', lb = lb, ub = ub)

        m.constrain(-100 <= x.sum() <= 100)

        self.assertEqual(m.minimize(x.sum()), -100)
        self.assertEqual(m[x[0]], -1)
        self.assertEqual(m[x[1]], -96)
        self.assertEqual(m[x[2]], -3)
        
        self.assertEqual(m.maximize(x.sum()), 100)
        self.assertEqual(m[x[0]], -1)
        self.assertEqual(m[x[1]], 104)
        self.assertEqual(m[x[2]], -3)

    def test05_rerun(self):

        m = CPlexModel()

        x = m.new()

        m.constrain(-1 <= x)
        m.constrain(x <= 1)

        self.assertEqual(m.minimize(x), -1)
        self.assertEqual(m.maximize(x), 1)

    def test06_linked_constraints(self):

        m = CPlexModel()

        x = m.new(name = 'x')

        m.constrain(-1 <= x <= 1)

        self.assertEqual(m.minimize(x), -1)
        self.assertEqual(m.maximize(x), 1)

    def test07_chaining_01(self):
        
        m = CPlexModel()

        x = m.new(name = 'x')
        y = m.new(name = 'y')

        m.constrain(0 <= x <= y <= 1)

        self.assertEqual(m.maximize(x+y), 2)
        self.assertEqual(m[x], 1)
        self.assertEqual(m[y], 1)

    def test07_chaining_02a(self):
        
        m = CPlexModel()

        x = m.new(name = 'x')

        m.constrain(x-1  <= 1)
        
        self.assertEqual(m.maximize(x), 2)

    def test07_chaining_02b(self):
        
        m = CPlexModel()

        x = m.new(name = 'x')
        y = m.new(name = 'y')

        m.constrain(x-1 <= y-1)
        m.constrain(y <= 5)

        self.assertEqual(m.maximize(x+y), 10)
        self.assertEqual(m[x], 5)
        self.assertEqual(m[y], 5)

    def test07_chaining_03(self):
        
        m = CPlexModel()

        x = m.new(name = 'x')
        y = m.new(name = 'y')
        z = m.new(name = 'z')

        m.constrain(0 <= x-1 <= y-2 <= z-3 <= 1)

        self.assertEqual(m.maximize(x+y+z), 2 + 3 + 4)
        self.assertEqual(m[x], 2)
        self.assertEqual(m[y], 3)
        self.assertEqual(m[z], 4)

    def test07_chaining_03_control(self):
        
        m = CPlexModel()

        x = m.new(name = 'x')
        y = m.new(name = 'y')
        z = m.new(name = 'z')

        m.constrain(0 <= x-1)
        m.constrain(x-1 <= y-2)
        m.constrain(y-2 <= z-3)
        m.constrain(z-3 <= 1)

        self.assertEqual(m.maximize(x+y+z), 2 + 3 + 4)
        self.assertEqual(m[x], 2)
        self.assertEqual(m[y], 3)
        self.assertEqual(m[z], 4)

    def test08_quadratic_objective(self):

        m = CPlexModel()

        x = m.new(size = 2, name = 'x', lb=1, ub=10)

        self.assertEqual(m.minimize(x[0]*x[0] + x[1]*x[1]), 2)
        self.assertEqual(m[x[0]], 1)
        self.assertEqual(m[x[1]], 1)

    def test09_abs_01(self):

        m = CPlexModel()

        x = m.new()

        m.constrain(x.abs() <= 1)

        self.assertEqual(m.maximize(x), 1)
        self.assertEqual(m.minimize(x), -1)

    def test09_abs_01b(self):

        m = CPlexModel()

        x = m.new()

        m.constrain(abs(x) <= 1)

        self.assertEqual(m.maximize(x), 1)
        self.assertEqual(m.minimize(x), -1)

    def test09_abs_02(self):

        m = CPlexModel()

        x = m.new()

        self.assertEqual(m.minimize( (x-1).abs() ), 0)
        self.assertEqual(m[x], 1)

    def test09_abs_03(self):

        m = CPlexModel()

        x = m.new(3)

        A = ar([[1,0,0], [1,1,0], [1,1,1]])
        y = ar([1,2,3])

        self.assertEqual(m.minimize( (A*x - y).abs().sum() ), 0)
        self.assertEqual(m[x[0]], 1)
        self.assertEqual(m[x[1]], 1)
        self.assertEqual(m[x[2]], 1)

    def test10_scalar_var_constrain_01(self):

        m = CPlexModel()

        x = m.new(2, lb = 0)
        t = m.new()

        m.constrain( x <= t)

        self.assertEqual(m.minimize(t), 0)
        self.assertEqual(m[x[0]], 0)

        
    def test10_scalar_var_constrain_02(self):

        m = CPlexModel()

        A = ar([[1,0,0], [1,1,0], [1,1,1]])
        y = ar([1,2,3])

        x = m.new(3)
        t = m.new()

        m.constrain( abs((A*x - y)) <= t)

        self.assertEqual(m.minimize(t), 0)
            
        self.assertEqual(m[x[0]], 1)
        self.assertEqual(m[x[1]], 1)
        self.assertEqual(m[x[2]], 1)
        

    def test11_value_retrieve_01(self):

        m = CPlexModel()
        x = m.new(2, name = 'x')
        m.constrain(eye(2) * x <= 1)
        m.maximize(x.sum())

        self.assert_( (x() == 1).all())
        self.assertEqual(x[0](), 1)
        self.assertEqual(x[1](), 1)

    def test11_value_retrieve_02(self):

        m = CPlexModel()
        x = m.new(2, name = 'x')
        m.constrain(eye(2) * x <= 1)
        m.maximize(x.sum())

        self.assert_( (x() == 1).all())
        self.assertEqual(x(0), 1)
        self.assertEqual(x(1), 1)

    def test12_value_set(self):

        m = CPlexModel()
        
        y = m.new()

        m.constrain(y <= 2)
        m.maximize(y, starting_dict = {y : 1} )

        self.assertEqual(m[y], 2)

    def test13_RecycleVars(self):
        # Get a big model together, see if starting it from the same
        # place speeds things up
        return

        m = CPlexModel()

        N = 200
        x = m.new(2*N, lb = 0, vtype=int)
        y = m.new(N, vtype = int)
        A = arange(2*N*N).reshape( (N, 2*N) )

        m.constrain(A*x <= y)
        m.constrain(N <= abs(x).sum() <= 2*N)

        m.minimize(y.sum())
        m.minimize(y.sum(), recycle_variables = True)

    def test14_mean(self):

        m = CPlexModel()

        x = m.new( (2,4) )

        m.constrain(x[:,:2].mean(1) == 1)
        m.constrain(x.mean(1) == 2)

        self.assertEqual(m.minimize(
            abs(x[:,:2] - x[:,:2].mean()).sum()
            + abs(x[:,2:] - x[:,2:].mean()).sum() ), 0)

        self.assert_( (m[x[:,:2]] == 1).all() )
        self.assert_( (m[x[:,2:]] == 3).all() )

    def test15_read_basis(self):

        f, name = tempfile.mkstemp(suffix='bas', prefix='tmp_cplex')
        try:

            m = CPlexModel()

            N = 1000
            x = m.new(2*N, lb = 0)
            y = m.new(N, lb = 0)
            A = rn.normal( size = (N, 2*N) )

            z = m.new()
            m.constrain(A*x <= y)
            m.constrain(x.sum() <= z*N)

            m.minimize(y.sum() + z, algorithm='primal')
            
            t1 = m.getSolverTime()
            print "no_basis, time = ", t1

            m.saveBasis(name)

            m.minimize(y.sum(), basis_file = name, algorithm='primal')
            t2 = m.getSolverTime()
            print "with basis, time = ", t2

            self.assert_(t2 < t1)
            
        finally:
            os.remove(name)
        
    def test16_recycle_basis(self):

        m = CPlexModel()

        N = 1000
        x = m.new(2*N, lb = 0)
        y = m.new(N, lb = 0)
        A = rn.normal( size = (N, 2*N) )

        z = m.new()
        m.constrain(A*x <= y)
        m.constrain(x.sum() <= z*N)

        m.minimize(y.sum() + z, algorithm='primal')

        t1 = m.getSolverTime()

        m.minimize(y.sum(), recycle_basis = True, algorithm='primal')
        t2 = m.getSolverTime()

        self.assert_(t2 < t1)

    def test17_remove_constraint(self):

        m = CPlexModel()
        x = m.new()
        y = m.new()

        m.constrain(0 <= x <= 5)
        m.constrain(0 <= y <= 5)
        c = (x <= 2)
        
        m.constrain(c)
        
        self.assert_(m.maximize(x + y) == 7)
        self.assert_(m[x] == 2)
        self.assert_(m[y] == 5)
        
        m.removeConstraint(c)
        
        self.assert_(m.maximize(x + y) == 10)
        self.assert_(m[x] == 5)
        self.assert_(m[y] == 5)

    def test20_model_infeasible(self):
        m = CPlexModel()
        x = m.new(lb = 0, ub = 10)
        m.constrain(x >= 12)

        self.assertRaises(CPlexNoSolution, lambda: m.maximize(x))

    def test21_model_unbounded(self):
        m = CPlexModel()
        x = m.new(lb = 0)
        m.constrain(x >= 12)

        self.assertRaises(CPlexNoSolution, lambda: m.maximize(x))
        

        


if __name__ == '__main__':
    unittest.main()

