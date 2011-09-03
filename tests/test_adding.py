from common import *

class TestMinimal(unittest.TestCase):

    def checkMinLP1(self, opts):
        
        lp = LP()

        indices = {}
        indices["t"] = (0,3)
        indices["n"] = "a"
        indices["N"] = "a"
        indices["l"] = [0,1,2]
        indices["a"] = ar([0,1,2])
        indices["r"] = ar([2,1,0])[::-1]
        indices["f"] = ar([0,1,2],dtype=float64)
        indices["e"] = None  # empty

        weights = {}
        weights["l"] = [1,1,1]
        weights["a"] = ar([1,1,1])
        weights["f"] = ar([1,1,1])
        weights["r"] = ar([1,1,1])[::-1]
        weights["s"] = 1

        obj_func = {}
        obj_func["l"] = [1,2,3]
        obj_func["a"] = ar([1,2,3])
        obj_func["f"] = ar([1,2,3],dtype=float64)
        obj_func["r"] = ar([3,2,1],dtype=float64)[::-1]
        obj_func["s"] = [1,2,3]  # can't do scalar here


        # Some ones used in the dict's case
        il = indices["l"] 
        assert len(il) == 3

        wl = weights["l"]
        assert len(wl) == 3

        ol = obj_func["l"]
        assert len(ol) == 3

        if opts[0] == "d" or opts[0] == "T":

            if opts[1] == "1":
                cd = [ (i, w) for i, w in zip(il, wl)]
                od = [ (i, o) for i, o in zip(il, ol)]

            elif opts[1] == "2":
                cd = [ ("a", wl[:2]), ("b", wl[2])]
                od = [ ("a", ol[:2]), ("b", ol[2])]
            
            elif opts[1] == "3":
                cd = [((0,2), wl[:2]), (2, wl[2])]
                od = [((0,2), ol[:2]), (2, ol[2])]

            elif opts[1] == "4":
                cd = [((0,2), wl[:2]), ( (2,3), wl[2])]
                od = [((0,2), ol[:2]), ( (2,3), ol[2])]

            elif opts[1] == "5":  # bad for out of order
                cd = [("a", wl[:2]), ( (2,3), wl[2])]
                od = [("a", ol[:2]), ( (2,3), ol[2])]
            
            elif opts[1] in indices.keys() and opts[2] in weights.keys():
                cd = [(indices[opts[1]], weights[opts[2]])]
                od = [(indices[opts[1]], obj_func[opts[2]])]
            else:
                assert False

            if opts[0] == "d":
                self.assert_(lp.addConstraint(dict(cd), ">=", 1) == 0)
                lp.setObjective(dict(od))
            elif opts[0] == "T":

                if opts[1] == "N":
                    lp.getIndexBlock(indices["N"], 3)

                self.assert_(lp.addConstraint(cd, ">=", 1) == 0)
                lp.setObjective(od)
        else:
            assert len(opts) == 2
            
            if opts[0] == "N":
                lp.getIndexBlock(indices["N"], 3)

            io = indices[opts[0]]

            if io is None:
                self.assert_(lp.addConstraint(weights[opts[1]], ">=", 1) == 0)
                lp.setObjective(obj_func[opts[1]])
            else:
                self.assert_(lp.addConstraint( (indices[opts[0]], weights[opts[1]]), ">=", 1) == 0)
                lp.setObjective( (indices[opts[0]], obj_func[opts[1]]))

        lp.solve()

        self.assertAlmostEqual(lp.getObjectiveValue(), 1)
        
        if opts[0] not in ["d", "T"]:
            v = lp.getSolution(indices[opts[0]])
        else:
            v = lp.getSolution()

        self.assert_(len(v) == 3, "len(v) = %d != 3" % len(v))

        self.assertAlmostEqual(lp.getSolution(0), 1)
        self.assertAlmostEqual(v[0], 1)
        self.assertAlmostEqual(lp.getSolution(1), 0)
        self.assertAlmostEqual(v[1], 0)
        self.assertAlmostEqual(lp.getSolution(2), 0)
        self.assertAlmostEqual(v[2], 0)


    def testConstraints_tl(self): self.checkMinLP1("tl")
    def testConstraints_ta(self): self.checkMinLP1("ta")
    def testConstraints_tf(self): self.checkMinLP1("tf")
    def testConstraints_ts(self): self.checkMinLP1("ts")
    def testConstraints_tr(self): self.checkMinLP1("tr")

    def testConstraints_nl(self): self.checkMinLP1("nl")
    def testConstraints_na(self): self.checkMinLP1("na")
    def testConstraints_nf(self): self.checkMinLP1("nf")
    def testConstraints_nr(self): self.checkMinLP1("nr")

    def testConstraints_Nl(self): self.checkMinLP1("Nl")
    def testConstraints_Na(self): self.checkMinLP1("Na")
    def testConstraints_Nf(self): self.checkMinLP1("Nf")
    def testConstraints_Ns(self): self.checkMinLP1("Ns")
    def testConstraints_Nr(self): self.checkMinLP1("Nr")

    def testConstraints_ll(self): self.checkMinLP1("ll")
    def testConstraints_la(self): self.checkMinLP1("la")
    def testConstraints_lf(self): self.checkMinLP1("lf")
    def testConstraints_ls(self): self.checkMinLP1("ls")
    def testConstraints_lr(self): self.checkMinLP1("lr")

    def testConstraints_al(self): self.checkMinLP1("al")
    def testConstraints_aa(self): self.checkMinLP1("aa")
    def testConstraints_af(self): self.checkMinLP1("af")
    def testConstraints_as(self): self.checkMinLP1("as")
    def testConstraints_ar(self): self.checkMinLP1("ar")

    def testConstraints_fl(self): self.checkMinLP1("fl")
    def testConstraints_fa(self): self.checkMinLP1("fa")
    def testConstraints_ff(self): self.checkMinLP1("ff")
    def testConstraints_fs(self): self.checkMinLP1("fs")
    def testConstraints_fr(self): self.checkMinLP1("fr")

    def testConstraints_el(self): self.checkMinLP1("el")
    def testConstraints_ea(self): self.checkMinLP1("ea")
    def testConstraints_ef(self): self.checkMinLP1("ef")
    def testConstraints_er(self): self.checkMinLP1("er")

    def testConstraints_rl(self): self.checkMinLP1("rl")
    def testConstraints_ra(self): self.checkMinLP1("ra")
    def testConstraints_rf(self): self.checkMinLP1("rf")
    def testConstraints_rs(self): self.checkMinLP1("rs")
    def testConstraints_rr(self): self.checkMinLP1("rr")


    def testConstraints_d1(self): self.checkMinLP1("d1")
    def testConstraints_d1(self): self.checkMinLP1("d1")
    def testConstraints_d1(self): self.checkMinLP1("d1")

    def testConstraints_d2(self): self.checkMinLP1("d2")
    def testConstraints_d2(self): self.checkMinLP1("d2")
    def testConstraints_d2(self): self.checkMinLP1("d2")

    def testConstraints_d3(self): self.checkMinLP1("d3")
    def testConstraints_d3(self): self.checkMinLP1("d3")
    def testConstraints_d3(self): self.checkMinLP1("d3")

    def testConstraints_d4(self): self.checkMinLP1("d4")
    def testConstraints_d4(self): self.checkMinLP1("d4")
    def testConstraints_d4(self): self.checkMinLP1("d4")

    def testConstraints_T1(self): self.checkMinLP1("T1")
    def testConstraints_T1(self): self.checkMinLP1("T1")
    def testConstraints_T1(self): self.checkMinLP1("T1")

    def testConstraints_T2(self): self.checkMinLP1("T2")
    def testConstraints_T2(self): self.checkMinLP1("T2")
    def testConstraints_T2(self): self.checkMinLP1("T2")

    def testConstraints_T3(self): self.checkMinLP1("T3")
    def testConstraints_T3(self): self.checkMinLP1("T3")
    def testConstraints_T3(self): self.checkMinLP1("T3")

    def testConstraints_T4(self): self.checkMinLP1("T4")
    def testConstraints_T4(self): self.checkMinLP1("T4")
    def testConstraints_T4(self): self.checkMinLP1("T4")

    def testConstraints_T5(self): self.checkMinLP1("T5")
    def testConstraints_T5(self): self.checkMinLP1("T5")
    def testConstraints_T5(self): self.checkMinLP1("T5")


    def testConstraints_Ttl(self): self.checkMinLP1("Ttl")
    def testConstraints_Tta(self): self.checkMinLP1("Tta")
    def testConstraints_Ttf(self): self.checkMinLP1("Ttf")
    def testConstraints_Tts(self): self.checkMinLP1("Tts")
    def testConstraints_Ttr(self): self.checkMinLP1("Ttr")

    def testConstraints_Tnl(self): self.checkMinLP1("Tnl")
    def testConstraints_Tna(self): self.checkMinLP1("Tna")
    def testConstraints_Tnf(self): self.checkMinLP1("Tnf")
    def testConstraints_Tnr(self): self.checkMinLP1("Tnr")

    def testConstraints_TNl(self): self.checkMinLP1("TNl")
    def testConstraints_TNa(self): self.checkMinLP1("TNa")
    def testConstraints_TNf(self): self.checkMinLP1("TNf")
    def testConstraints_TNs(self): self.checkMinLP1("TNs")
    def testConstraints_TNr(self): self.checkMinLP1("TNr")

    def testConstraints_Tll(self): self.checkMinLP1("Tll")
    def testConstraints_Tla(self): self.checkMinLP1("Tla")
    def testConstraints_Tlf(self): self.checkMinLP1("Tlf")
    def testConstraints_Tls(self): self.checkMinLP1("Tls")
    def testConstraints_Tlr(self): self.checkMinLP1("Tlr")

    def testConstraints_Tal(self): self.checkMinLP1("Tal")
    def testConstraints_Taa(self): self.checkMinLP1("Taa")
    def testConstraints_Taf(self): self.checkMinLP1("Taf")
    def testConstraints_Tas(self): self.checkMinLP1("Tas")
    def testConstraints_Tar(self): self.checkMinLP1("Tar")

    def testConstraints_Tfl(self): self.checkMinLP1("Tfl")
    def testConstraints_Tfa(self): self.checkMinLP1("Tfa")
    def testConstraints_Tff(self): self.checkMinLP1("Tff")
    def testConstraints_Tfs(self): self.checkMinLP1("Tfs")
    def testConstraints_Tfr(self): self.checkMinLP1("Tfr")

    def testConstraints_Tel(self): self.checkMinLP1("Tel")
    def testConstraints_Tea(self): self.checkMinLP1("Tea")
    def testConstraints_Tef(self): self.checkMinLP1("Tef")
    def testConstraints_Ter(self): self.checkMinLP1("Ter")

    def testConstraints_Trl(self): self.checkMinLP1("Trl")
    def testConstraints_Tra(self): self.checkMinLP1("Tra")
    def testConstraints_Trf(self): self.checkMinLP1("Trf")
    def testConstraints_Trs(self): self.checkMinLP1("Trs")
    def testConstraints_Trr(self): self.checkMinLP1("Trr")



class TestTwoLevel(unittest.TestCase):

    def checkMinLP1(self, opts):
        
        lp = LP()

        idxlist = [{}, {}]

        idxlist[0]["t"] = (0,3)
        idxlist[0]["n"] = "a"
        idxlist[0]["N"] = "a"
        idxlist[0]["l"] = [0,1,2]
        idxlist[0]["a"] = ar([0,1,2])
        idxlist[0]["r"] = ar([0,0,1,1,2,2])[::2]
        idxlist[0]["f"] = ar([0,1,2],dtype=float64)
        idxlist[0]["e"] = None  # empty

        idxlist[1]["t"] = (3,6)
        idxlist[1]["n"] = "b"
        idxlist[1]["N"] = "b"
        idxlist[1]["l"] = [3,4,5]
        idxlist[1]["a"] = ar([3,4,5])
        idxlist[1]["r"] = ar([3,3,4,4,5,5])[::2]
        idxlist[1]["f"] = ar([3,4,5],dtype=float64)
        idxlist[1]["e"] = (3,6)

        weightlist = [{}, {}]
        weightlist[0]["l"] = [1,1,1]
        weightlist[0]["a"] = ar([1,1,1])
        weightlist[0]["f"] = ar([1,1,1])
        weightlist[0]["r"] = ar([1,1,1,1,1,1])[::2]
        weightlist[0]["s"] = 1

        weightlist[1]["l"] = [1,0.5,0.5]
        weightlist[1]["a"] = ar([1,0.5,0.5])
        weightlist[1]["f"] = ar([1,0.5,0.5])
        weightlist[1]["r"] = ar([1,1,0.5,0.5,0.5,0.5])[::2]
        weightlist[1]["s"] = [1.0, 0.5, 0.5]

        obj_func_list = [{},{}]
        obj_func_list[0]["l"] = [1,2,3]
        obj_func_list[0]["a"] = ar([1,2,3])
        obj_func_list[0]["f"] = ar([1,2,3],dtype=float64)
        obj_func_list[0]["r"] = ar([1,1,2,2,3,3],dtype=float64)[::2]
        obj_func_list[0]["s"] = [1,2,3]  # can't do scalar here

        obj_func_list[1]["l"] = [1,1,1]
        obj_func_list[1]["a"] = ar([1,1,1])
        obj_func_list[1]["f"] = ar([1,1,1],dtype=float64)
        obj_func_list[1]["r"] = ar([1,1,1,1,1,1],dtype=float64)[::2]
        obj_func_list[1]["s"] = 1

        register_check = {}
        disable_regular_check = False

        for ci, (indices, weights, obj_func) in enumerate(zip(idxlist, weightlist, obj_func_list)):

            # Some ones used in the dict's case
            il = indices["l"]
            assert len(il) == 3

            wl = weights["l"]
            assert len(wl) == 3

            ol = obj_func["l"]
            assert len(ol) == 3

            if opts[0] == "d" or opts[0] == "T":

                t = il[0]
                assert il[-1] - il[0] == 2

                n1 = indices["n"]
                n2 = indices["n"]+"2"

                if opts[1] == "1":
                    cd = [ (i, w) for i, w in zip(il, wl)]
                    od = [ (i, o) for i, o in zip(il, ol)]

                elif opts[1] == "2":
                    cd = [ (n1, wl[:2]), (n2, wl[2])]
                    od = [ (n1, ol[:2]), (n2, ol[2])]
                    
                    register_check[n1] = [1,0]
                    register_check[n2] = [0]
                    disable_regular_check = True

                elif opts[1] == "3":
                    cd = [((t,t+2), wl[:2]), (t+2, wl[2])]
                    od = [((t,t+2), ol[:2]), (t+2, wl[2])]

                elif opts[1] == "4":
                    cd = [((t,t+2), wl[:2]), ((t+2,t+3), wl[2])]
                    od = [((t,t+2), ol[:2]), ((t+2,t+3), wl[2])]

                elif opts[1] == "5":  # bad for dict
                    cd = [(n1, wl[:2]), ((t+2,t+3), wl[2])]
                    od = [(n1, ol[:2]), ((t+2,t+3), wl[2])]

                elif opts[1] in indices.keys() and opts[2] in weights.keys():
                    cd = [(indices[opts[1]], weights[opts[2]])]
                    od = [(indices[opts[1]], obj_func[opts[2]])]

                else:
                    assert False

                if opts[0] == "d":
                    self.assert_(lp.addConstraint(dict(cd), ">=", 1) == ci)
                    lp.addToObjective(dict(od))
                elif opts[0] == "T":

                    if opts[1] == "N":
                        lp.getIndexBlock(indices["N"], 3)

                    self.assert_(lp.addConstraint(cd, ">=", 1) == ci)
                    lp.addToObjective(od)
            else:
                assert len(opts) == 2

                if opts[0] == "N":
                    lp.getIndexBlock(indices["N"], 3)

                io = indices[opts[0]]

                if io is None:
                    self.assert_(lp.addConstraint(weights[opts[1]], ">=", 1) == ci)
                    lp.addToObjective(obj_func[opts[1]])
                else:
                    self.assert_(lp.addConstraint( (indices[opts[0]], weights[opts[1]]), ">=", 1) == ci)
                    lp.addToObjective( (indices[opts[0]], obj_func[opts[1]]))

        for num_times in range(2):
            lp.solve()

            self.assertAlmostEqual(lp.getObjectiveValue(), 2)

            if disable_regular_check:
                for k, l in register_check.iteritems():
                    v = lp.getSolution(k)
                    self.assert_(len(v) == len(l))
                    for i1,i2 in zip(l,v):
                        self.assertAlmostEqual(i1,i2)
            else:
                v = lp.getSolution()

                self.assert_(len(v) == 6, "len(v) = %d != 6" % len(v))
                self.assertAlmostEqual(v[0], 1)
                self.assertAlmostEqual(v[1], 0)
                self.assertAlmostEqual(v[2], 0)
                self.assertAlmostEqual(v[3], 1)
                self.assertAlmostEqual(v[4], 0)
                self.assertAlmostEqual(v[5], 0)

                if opts[0] in "nN":

                    d = lp.getSolutionDict()

                    self.assert_(set(d.iterkeys()) == set(["a", "b"]))

                    self.assertAlmostEqual(d["a"][0], 1)
                    self.assertAlmostEqual(d["a"][1], 0)
                    self.assertAlmostEqual(d["a"][2], 0)
                    self.assertAlmostEqual(d["b"][0], 1)
                    self.assertAlmostEqual(d["b"][1], 0)
                    self.assertAlmostEqual(d["b"][2], 0)

            # now test the retrieval

    def testConstraints_tl(self): self.checkMinLP1("tl")
    def testConstraints_ta(self): self.checkMinLP1("ta")
    def testConstraints_tf(self): self.checkMinLP1("tf")
    def testConstraints_ts(self): self.checkMinLP1("ts")
    def testConstraints_tr(self): self.checkMinLP1("tr")

    def testConstraints_nl(self): self.checkMinLP1("nl")
    def testConstraints_na(self): self.checkMinLP1("na")
    def testConstraints_nf(self): self.checkMinLP1("nf")
    def testConstraints_nr(self): self.checkMinLP1("nr")

    def testConstraints_Nl(self): self.checkMinLP1("Nl")
    def testConstraints_Na(self): self.checkMinLP1("Na")
    def testConstraints_Nf(self): self.checkMinLP1("Nf")
    def testConstraints_Ns(self): self.checkMinLP1("Ns")
    def testConstraints_Nr(self): self.checkMinLP1("Nr")

    def testConstraints_ll(self): self.checkMinLP1("ll")
    def testConstraints_la(self): self.checkMinLP1("la")
    def testConstraints_lf(self): self.checkMinLP1("lf")
    def testConstraints_ls(self): self.checkMinLP1("ls")
    def testConstraints_lr(self): self.checkMinLP1("lr")

    def testConstraints_al(self): self.checkMinLP1("al")
    def testConstraints_aa(self): self.checkMinLP1("aa")
    def testConstraints_af(self): self.checkMinLP1("af")
    def testConstraints_as(self): self.checkMinLP1("as")
    def testConstraints_ar(self): self.checkMinLP1("ar")

    def testConstraints_fl(self): self.checkMinLP1("fl")
    def testConstraints_fa(self): self.checkMinLP1("fa")
    def testConstraints_ff(self): self.checkMinLP1("ff")
    def testConstraints_fs(self): self.checkMinLP1("fs")
    def testConstraints_fr(self): self.checkMinLP1("fr")

    def testConstraints_el(self): self.checkMinLP1("el")
    def testConstraints_ea(self): self.checkMinLP1("ea")
    def testConstraints_ef(self): self.checkMinLP1("ef")
    def testConstraints_er(self): self.checkMinLP1("er")

    def testConstraints_rl(self): self.checkMinLP1("rl")
    def testConstraints_ra(self): self.checkMinLP1("ra")
    def testConstraints_rf(self): self.checkMinLP1("rf")
    def testConstraints_rs(self): self.checkMinLP1("rs")
    def testConstraints_rr(self): self.checkMinLP1("rr")


    def testConstraints_d1(self): self.checkMinLP1("d1")
    def testConstraints_d1(self): self.checkMinLP1("d1")
    def testConstraints_d1(self): self.checkMinLP1("d1")

    def testConstraints_d2(self): self.checkMinLP1("d2")
    def testConstraints_d2(self): self.checkMinLP1("d2")
    def testConstraints_d2(self): self.checkMinLP1("d2")

    def testConstraints_d3(self): self.checkMinLP1("d3")
    def testConstraints_d3(self): self.checkMinLP1("d3")
    def testConstraints_d3(self): self.checkMinLP1("d3")

    def testConstraints_d4(self): self.checkMinLP1("d4")
    def testConstraints_d4(self): self.checkMinLP1("d4")
    def testConstraints_d4(self): self.checkMinLP1("d4")

    def testConstraints_T1(self): self.checkMinLP1("T1")
    def testConstraints_T1(self): self.checkMinLP1("T1")
    def testConstraints_T1(self): self.checkMinLP1("T1")

    def testConstraints_T2(self): self.checkMinLP1("T2")
    def testConstraints_T2(self): self.checkMinLP1("T2")
    def testConstraints_T2(self): self.checkMinLP1("T2")

    def testConstraints_T3(self): self.checkMinLP1("T3")
    def testConstraints_T3(self): self.checkMinLP1("T3")
    def testConstraints_T3(self): self.checkMinLP1("T3")

    def testConstraints_T4(self): self.checkMinLP1("T4")
    def testConstraints_T4(self): self.checkMinLP1("T4")
    def testConstraints_T4(self): self.checkMinLP1("T4")

    def testConstraints_T5(self): self.checkMinLP1("T5")
    def testConstraints_T5(self): self.checkMinLP1("T5")
    def testConstraints_T5(self): self.checkMinLP1("T5")


    def testConstraints_Ttl(self): self.checkMinLP1("Ttl")
    def testConstraints_Tta(self): self.checkMinLP1("Tta")
    def testConstraints_Ttf(self): self.checkMinLP1("Ttf")
    def testConstraints_Tts(self): self.checkMinLP1("Tts")
    def testConstraints_Ttr(self): self.checkMinLP1("Ttr")

    def testConstraints_Tnl(self): self.checkMinLP1("Tnl")
    def testConstraints_Tna(self): self.checkMinLP1("Tna")
    def testConstraints_Tnf(self): self.checkMinLP1("Tnf")
    def testConstraints_Tnr(self): self.checkMinLP1("Tnr")

    def testConstraints_TNl(self): self.checkMinLP1("TNl")
    def testConstraints_TNa(self): self.checkMinLP1("TNa")
    def testConstraints_TNf(self): self.checkMinLP1("TNf")
    def testConstraints_TNs(self): self.checkMinLP1("TNs")
    def testConstraints_TNr(self): self.checkMinLP1("TNr")

    def testConstraints_Tll(self): self.checkMinLP1("Tll")
    def testConstraints_Tla(self): self.checkMinLP1("Tla")
    def testConstraints_Tlf(self): self.checkMinLP1("Tlf")
    def testConstraints_Tls(self): self.checkMinLP1("Tls")
    def testConstraints_Tlr(self): self.checkMinLP1("Tlr")

    def testConstraints_Tal(self): self.checkMinLP1("Tal")
    def testConstraints_Taa(self): self.checkMinLP1("Taa")
    def testConstraints_Taf(self): self.checkMinLP1("Taf")
    def testConstraints_Tas(self): self.checkMinLP1("Tas")
    def testConstraints_Tar(self): self.checkMinLP1("Tar")

    def testConstraints_Tfl(self): self.checkMinLP1("Tfl")
    def testConstraints_Tfa(self): self.checkMinLP1("Tfa")
    def testConstraints_Tff(self): self.checkMinLP1("Tff")
    def testConstraints_Tfs(self): self.checkMinLP1("Tfs")
    def testConstraints_Tfr(self): self.checkMinLP1("Tfr")

    def testConstraints_Tel(self): self.checkMinLP1("Tel")
    def testConstraints_Tea(self): self.checkMinLP1("Tea")
    def testConstraints_Tef(self): self.checkMinLP1("Tef")
    def testConstraints_Ter(self): self.checkMinLP1("Ter")

    def testConstraints_Trl(self): self.checkMinLP1("Trl")
    def testConstraints_Tra(self): self.checkMinLP1("Tra")
    def testConstraints_Trf(self): self.checkMinLP1("Trf")
    def testConstraints_Trs(self): self.checkMinLP1("Trs")
    def testConstraints_Trr(self): self.checkMinLP1("Trr")
            


    ############################################################
    # Test 2d stuff
class Test2dMatrix(unittest.TestCase):

    def check2dMatrix(self, opts):

        values = {}

        indices = {}
        indices["t"] = (0,3)
        indices["n"] = "a"
        indices["N"] = "a"
        indices["l"] = [0,1,2]
        indices["a"] = ar([0,1,2],dtype=uint)
        indices["r"] = ar([0,0,1,1,2,2],dtype=uint)[::2]
        indices["f"] = ar([0,1,2],dtype=float64)
        indices["e"] = None  # empty

        A = [[1,0,  0],
             [0,0,  0.5],
             [0,0.5,0]]

        values = {}
        values["L"] = A
        values["l"] = [ar(le) for le in A]
        values["a"] = ar(A)

        targets = {}
        targets["s"] = 1
        targets["l"] = [1,1,1]
        targets["a"] = ar([1,1,1],dtype=uint)
        targets["r"] = ar([1,1,1,1,1,1],dtype=uint)[::2]
        targets["f"] = ar([1,1,1],dtype=float64)

        targets_t = {}
        targets_t["t"] = tuple
        targets_t["l"] = list

        targets_u = {}
        targets_u["s"] = 10
        targets_u["l"] = [10,10,10]
        targets_u["a"] = ar([10,10,10],dtype=uint)
        targets_u["r"] = ar([10,10,10,10,10,10],dtype=uint)[::2]
        targets_u["f"] = ar([10,10,10],dtype=float64)

        lp = LP()

        if opts[0] == "N":
            lp.getIndexBlock(indices["N"], 3)

        io = indices[opts[0]]
        vl = values [opts[1]]
        
        if len(opts) == 3:
            tr = targets[opts[2]]
            cstr = ">="
        else:
            tr = targets_t[opts[3]]([targets[opts[2]], targets[opts[4]]])
            cstr = "in"

        ob = [1,2,3]
        
        c_ret_idx = [0,1,2]

        if io is None:
            ret = lp.addConstraint(vl, cstr, tr)
            self.assert_(ret == c_ret_idx, "%s != %s" %(str(ret), str(c_ret_idx)))
            lp.setObjective(ob)
        else:
            ret = lp.addConstraint( (io, vl), cstr, tr)
            self.assert_(ret == c_ret_idx, "%s != %s" %(str(ret), str(c_ret_idx)))
            lp.setObjective( (io, ob) )

        for num_times in range(2):  # make sure it's same anser second time
            lp.solve()

            self.assertAlmostEqual(lp.getObjectiveValue(), 11)

            v = lp.getSolution()

            self.assert_(len(v) == 3)
            self.assertAlmostEqual(v[0], 1)
            self.assertAlmostEqual(v[1], 2)
            self.assertAlmostEqual(v[2], 2)


    def test2DMatrix_tLs(self): self.check2dMatrix("tLs")
    def test2DMatrix_tLl(self): self.check2dMatrix("tLl")
    def test2DMatrix_tLa(self): self.check2dMatrix("tLa")
    def test2DMatrix_tLf(self): self.check2dMatrix("tLf")
    def test2DMatrix_tLr(self): self.check2dMatrix("tLr")

    def test2DMatrix_tls(self): self.check2dMatrix("tls")
    def test2DMatrix_tll(self): self.check2dMatrix("tll")
    def test2DMatrix_tla(self): self.check2dMatrix("tla")
    def test2DMatrix_tlf(self): self.check2dMatrix("tlf")
    def test2DMatrix_tlr(self): self.check2dMatrix("tlr")

    def test2DMatrix_tas(self): self.check2dMatrix("tas")
    def test2DMatrix_tal(self): self.check2dMatrix("tal")
    def test2DMatrix_taa(self): self.check2dMatrix("taa")
    def test2DMatrix_taf(self): self.check2dMatrix("taf")
    def test2DMatrix_tar(self): self.check2dMatrix("tar")

    
    def test2DMatrix_nLs(self): self.check2dMatrix("nLs")
    def test2DMatrix_nLl(self): self.check2dMatrix("nLl")
    def test2DMatrix_nLa(self): self.check2dMatrix("nLa")
    def test2DMatrix_nLf(self): self.check2dMatrix("nLf")
    def test2DMatrix_nLr(self): self.check2dMatrix("nLr")

    def test2DMatrix_nls(self): self.check2dMatrix("nls")
    def test2DMatrix_nll(self): self.check2dMatrix("nll")
    def test2DMatrix_nla(self): self.check2dMatrix("nla")
    def test2DMatrix_nlf(self): self.check2dMatrix("nlf")
    def test2DMatrix_nlr(self): self.check2dMatrix("nlr")

    def test2DMatrix_nas(self): self.check2dMatrix("nas")
    def test2DMatrix_nal(self): self.check2dMatrix("nal")
    def test2DMatrix_naa(self): self.check2dMatrix("naa")
    def test2DMatrix_naf(self): self.check2dMatrix("naf")


    def test2DMatrix_NLs(self): self.check2dMatrix("NLs")
    def test2DMatrix_NLl(self): self.check2dMatrix("NLl")
    def test2DMatrix_NLa(self): self.check2dMatrix("NLa")
    def test2DMatrix_NLf(self): self.check2dMatrix("NLf")
    def test2DMatrix_NLr(self): self.check2dMatrix("NLr")

    def test2DMatrix_Nls(self): self.check2dMatrix("Nls")
    def test2DMatrix_Nll(self): self.check2dMatrix("Nll")
    def test2DMatrix_Nla(self): self.check2dMatrix("Nla")
    def test2DMatrix_Nlf(self): self.check2dMatrix("Nlf")
    def test2DMatrix_Nlr(self): self.check2dMatrix("Nlr")

    def test2DMatrix_Nas(self): self.check2dMatrix("Nas")
    def test2DMatrix_Nal(self): self.check2dMatrix("Nal")
    def test2DMatrix_Naa(self): self.check2dMatrix("Naa")
    def test2DMatrix_Naf(self): self.check2dMatrix("Naf")
    def test2DMatrix_Nar(self): self.check2dMatrix("Nar")


    def test2DMatrix_lLs(self): self.check2dMatrix("lLs")
    def test2DMatrix_lLl(self): self.check2dMatrix("lLl")
    def test2DMatrix_lLa(self): self.check2dMatrix("lLa")
    def test2DMatrix_lLf(self): self.check2dMatrix("lLf")
    def test2DMatrix_lLr(self): self.check2dMatrix("lLr")

    def test2DMatrix_lls(self): self.check2dMatrix("lls")
    def test2DMatrix_lll(self): self.check2dMatrix("lll")
    def test2DMatrix_lla(self): self.check2dMatrix("lla")
    def test2DMatrix_llf(self): self.check2dMatrix("llf")
    def test2DMatrix_llr(self): self.check2dMatrix("llr")

    def test2DMatrix_las(self): self.check2dMatrix("las")
    def test2DMatrix_lal(self): self.check2dMatrix("lal")
    def test2DMatrix_laa(self): self.check2dMatrix("laa")
    def test2DMatrix_laf(self): self.check2dMatrix("laf")
    def test2DMatrix_lar(self): self.check2dMatrix("lar")


    def test2DMatrix_aLs(self): self.check2dMatrix("aLs")
    def test2DMatrix_aLl(self): self.check2dMatrix("aLl")
    def test2DMatrix_aLa(self): self.check2dMatrix("aLa")
    def test2DMatrix_aLf(self): self.check2dMatrix("aLf")
    def test2DMatrix_aLr(self): self.check2dMatrix("aLr")

    def test2DMatrix_als(self): self.check2dMatrix("als")
    def test2DMatrix_all(self): self.check2dMatrix("all")
    def test2DMatrix_ala(self): self.check2dMatrix("ala")
    def test2DMatrix_alf(self): self.check2dMatrix("alf")
    def test2DMatrix_alr(self): self.check2dMatrix("alr")

    def test2DMatrix_aas(self): self.check2dMatrix("aas")
    def test2DMatrix_aal(self): self.check2dMatrix("aal")
    def test2DMatrix_aaa(self): self.check2dMatrix("aaa")
    def test2DMatrix_aaf(self): self.check2dMatrix("aaf")
    def test2DMatrix_aar(self): self.check2dMatrix("aar")


    def test2DMatrix_fLs(self): self.check2dMatrix("fLs")
    def test2DMatrix_fLl(self): self.check2dMatrix("fLl")
    def test2DMatrix_fLa(self): self.check2dMatrix("fLa")
    def test2DMatrix_fLf(self): self.check2dMatrix("fLf")
    def test2DMatrix_fLr(self): self.check2dMatrix("fLr")

    def test2DMatrix_fls(self): self.check2dMatrix("fls")
    def test2DMatrix_fll(self): self.check2dMatrix("fll")
    def test2DMatrix_fla(self): self.check2dMatrix("fla")
    def test2DMatrix_flf(self): self.check2dMatrix("flf")
    def test2DMatrix_flr(self): self.check2dMatrix("flr")

    def test2DMatrix_fas(self): self.check2dMatrix("fas")
    def test2DMatrix_fal(self): self.check2dMatrix("fal")
    def test2DMatrix_faa(self): self.check2dMatrix("faa")
    def test2DMatrix_faf(self): self.check2dMatrix("faf")
    def test2DMatrix_far(self): self.check2dMatrix("far")


    def test2DMatrix_eLs(self): self.check2dMatrix("eLs")
    def test2DMatrix_eLl(self): self.check2dMatrix("eLl")
    def test2DMatrix_eLa(self): self.check2dMatrix("eLa")
    def test2DMatrix_eLf(self): self.check2dMatrix("eLf")
    def test2DMatrix_eLr(self): self.check2dMatrix("eLr")

    def test2DMatrix_els(self): self.check2dMatrix("els")
    def test2DMatrix_ell(self): self.check2dMatrix("ell")
    def test2DMatrix_ela(self): self.check2dMatrix("ela")
    def test2DMatrix_elf(self): self.check2dMatrix("elf")
    def test2DMatrix_elr(self): self.check2dMatrix("elr")

    def test2DMatrix_eas(self): self.check2dMatrix("eas")
    def test2DMatrix_eal(self): self.check2dMatrix("eal")
    def test2DMatrix_eaa(self): self.check2dMatrix("eaa")
    def test2DMatrix_eaf(self): self.check2dMatrix("eaf")
    def test2DMatrix_ear(self): self.check2dMatrix("ear")


    def test2DMatrix_rLs(self): self.check2dMatrix("rLs")
    def test2DMatrix_rLl(self): self.check2dMatrix("rLl")
    def test2DMatrix_rLa(self): self.check2dMatrix("rLa")
    def test2DMatrix_rLf(self): self.check2dMatrix("rLf")
    def test2DMatrix_rLr(self): self.check2dMatrix("rLr")

    def test2DMatrix_rls(self): self.check2dMatrix("rls")
    def test2DMatrix_rll(self): self.check2dMatrix("rll")
    def test2DMatrix_rla(self): self.check2dMatrix("rla")
    def test2DMatrix_rlf(self): self.check2dMatrix("rlf")
    def test2DMatrix_rlr(self): self.check2dMatrix("rlr")

    def test2DMatrix_ras(self): self.check2dMatrix("ras")
    def test2DMatrix_ral(self): self.check2dMatrix("ral")
    def test2DMatrix_raa(self): self.check2dMatrix("raa")
    def test2DMatrix_raf(self): self.check2dMatrix("raf")
    def test2DMatrix_rar(self): self.check2dMatrix("rar")


    # Test a restricted set to look at the bounds; namely fix the tuple at the beginning
    def test2DMatrix_tLala(self): self.check2dMatrix("tLala")
    def test2DMatrix_tLalf(self): self.check2dMatrix("tLalf")
    def test2DMatrix_tLall(self): self.check2dMatrix("tLall")
    def test2DMatrix_tLals(self): self.check2dMatrix("tLals")
    def test2DMatrix_tLalr(self): self.check2dMatrix("tLalr")

    def test2DMatrix_tLata(self): self.check2dMatrix("tLata")
    def test2DMatrix_tLatf(self): self.check2dMatrix("tLatf")
    def test2DMatrix_tLatl(self): self.check2dMatrix("tLatl")
    def test2DMatrix_tLats(self): self.check2dMatrix("tLats")
    def test2DMatrix_tLatr(self): self.check2dMatrix("tLatr")


    def test2DMatrix_tLfla(self): self.check2dMatrix("tLfla")
    def test2DMatrix_tLflf(self): self.check2dMatrix("tLflf")
    def test2DMatrix_tLfll(self): self.check2dMatrix("tLfll")
    def test2DMatrix_tLfls(self): self.check2dMatrix("tLfls")
    def test2DMatrix_tLflr(self): self.check2dMatrix("tLflr")

    def test2DMatrix_tLfta(self): self.check2dMatrix("tLfta")
    def test2DMatrix_tLftf(self): self.check2dMatrix("tLftf")
    def test2DMatrix_tLftl(self): self.check2dMatrix("tLftl")
    def test2DMatrix_tLfts(self): self.check2dMatrix("tLfts")
    def test2DMatrix_tLftr(self): self.check2dMatrix("tLftr")


    def test2DMatrix_tLlla(self): self.check2dMatrix("tLlla")
    def test2DMatrix_tLllf(self): self.check2dMatrix("tLllf")
    def test2DMatrix_tLlll(self): self.check2dMatrix("tLlll")
    def test2DMatrix_tLlls(self): self.check2dMatrix("tLlls")
    def test2DMatrix_tLllr(self): self.check2dMatrix("tLllr")

    def test2DMatrix_tLlta(self): self.check2dMatrix("tLlta")
    def test2DMatrix_tLltf(self): self.check2dMatrix("tLltf")
    def test2DMatrix_tLltl(self): self.check2dMatrix("tLltl")
    def test2DMatrix_tLlts(self): self.check2dMatrix("tLlts")
    def test2DMatrix_tLltr(self): self.check2dMatrix("tLltr")


    def test2DMatrix_tLsla(self): self.check2dMatrix("tLsla")
    def test2DMatrix_tLslf(self): self.check2dMatrix("tLslf")
    def test2DMatrix_tLsll(self): self.check2dMatrix("tLsll")
    def test2DMatrix_tLsls(self): self.check2dMatrix("tLsls")
    def test2DMatrix_tLslr(self): self.check2dMatrix("tLslr")

    def test2DMatrix_tLsta(self): self.check2dMatrix("tLsta")
    def test2DMatrix_tLstf(self): self.check2dMatrix("tLstf")
    def test2DMatrix_tLstl(self): self.check2dMatrix("tLstl")
    def test2DMatrix_tLsts(self): self.check2dMatrix("tLsts")
    def test2DMatrix_tLstr(self): self.check2dMatrix("tLstr")


    def test2DMatrix_tLrla(self): self.check2dMatrix("tLrla")
    def test2DMatrix_tLrlf(self): self.check2dMatrix("tLrlf")
    def test2DMatrix_tLrll(self): self.check2dMatrix("tLrll")
    def test2DMatrix_tLrls(self): self.check2dMatrix("tLrls")
    def test2DMatrix_tLrlr(self): self.check2dMatrix("tLrlr")

    def test2DMatrix_tLrta(self): self.check2dMatrix("tLrta")
    def test2DMatrix_tLrtf(self): self.check2dMatrix("tLrtf")
    def test2DMatrix_tLrtl(self): self.check2dMatrix("tLrtl")
    def test2DMatrix_tLrts(self): self.check2dMatrix("tLrts")
    def test2DMatrix_tLrtr(self): self.check2dMatrix("tLrtr")


    def test2DMatrix_taala(self): self.check2dMatrix("taala")
    def test2DMatrix_taalf(self): self.check2dMatrix("taalf")
    def test2DMatrix_taall(self): self.check2dMatrix("taall")
    def test2DMatrix_taals(self): self.check2dMatrix("taals")
    def test2DMatrix_taalr(self): self.check2dMatrix("taalr")

    def test2DMatrix_taata(self): self.check2dMatrix("taata")
    def test2DMatrix_taatf(self): self.check2dMatrix("taatf")
    def test2DMatrix_taatl(self): self.check2dMatrix("taatl")
    def test2DMatrix_taats(self): self.check2dMatrix("taats")
    def test2DMatrix_taatr(self): self.check2dMatrix("taatr")


    def test2DMatrix_tafla(self): self.check2dMatrix("tafla")
    def test2DMatrix_taflf(self): self.check2dMatrix("taflf")
    def test2DMatrix_tafll(self): self.check2dMatrix("tafll")
    def test2DMatrix_tafls(self): self.check2dMatrix("tafls")
    def test2DMatrix_taflr(self): self.check2dMatrix("taflr")

    def test2DMatrix_tafta(self): self.check2dMatrix("tafta")
    def test2DMatrix_taftf(self): self.check2dMatrix("taftf")
    def test2DMatrix_taftl(self): self.check2dMatrix("taftl")
    def test2DMatrix_tafts(self): self.check2dMatrix("tafts")
    def test2DMatrix_taftr(self): self.check2dMatrix("taftr")


    def test2DMatrix_talla(self): self.check2dMatrix("talla")
    def test2DMatrix_tallf(self): self.check2dMatrix("tallf")
    def test2DMatrix_talll(self): self.check2dMatrix("talll")
    def test2DMatrix_talls(self): self.check2dMatrix("talls")
    def test2DMatrix_tallr(self): self.check2dMatrix("tallr")

    def test2DMatrix_talta(self): self.check2dMatrix("talta")
    def test2DMatrix_taltf(self): self.check2dMatrix("taltf")
    def test2DMatrix_taltl(self): self.check2dMatrix("taltl")
    def test2DMatrix_talts(self): self.check2dMatrix("talts")
    def test2DMatrix_taltr(self): self.check2dMatrix("taltr")


    def test2DMatrix_tasla(self): self.check2dMatrix("tasla")
    def test2DMatrix_taslf(self): self.check2dMatrix("taslf")
    def test2DMatrix_tasll(self): self.check2dMatrix("tasll")
    def test2DMatrix_tasls(self): self.check2dMatrix("tasls")
    def test2DMatrix_taslr(self): self.check2dMatrix("taslr")

    def test2DMatrix_tasta(self): self.check2dMatrix("tasta")
    def test2DMatrix_tastf(self): self.check2dMatrix("tastf")
    def test2DMatrix_tastl(self): self.check2dMatrix("tastl")
    def test2DMatrix_tasts(self): self.check2dMatrix("tasts")
    def test2DMatrix_tastr(self): self.check2dMatrix("tastr")


    def test2DMatrix_tarla(self): self.check2dMatrix("tarla")
    def test2DMatrix_tarlf(self): self.check2dMatrix("tarlf")
    def test2DMatrix_tarll(self): self.check2dMatrix("tarll")
    def test2DMatrix_tarls(self): self.check2dMatrix("tarls")
    def test2DMatrix_tarlr(self): self.check2dMatrix("tarlr")

    def test2DMatrix_tarta(self): self.check2dMatrix("tarta")
    def test2DMatrix_tartf(self): self.check2dMatrix("tartf")
    def test2DMatrix_tartl(self): self.check2dMatrix("tartl")
    def test2DMatrix_tarts(self): self.check2dMatrix("tarts")
    def test2DMatrix_tartr(self): self.check2dMatrix("tartr")


    def test2DMatrix_tlala(self): self.check2dMatrix("tlala")
    def test2DMatrix_tlalf(self): self.check2dMatrix("tlalf")
    def test2DMatrix_tlall(self): self.check2dMatrix("tlall")
    def test2DMatrix_tlals(self): self.check2dMatrix("tlals")
    def test2DMatrix_tlalr(self): self.check2dMatrix("tlalr")

    def test2DMatrix_tlata(self): self.check2dMatrix("tlata")
    def test2DMatrix_tlatf(self): self.check2dMatrix("tlatf")
    def test2DMatrix_tlatl(self): self.check2dMatrix("tlatl")
    def test2DMatrix_tlats(self): self.check2dMatrix("tlats")
    def test2DMatrix_tlatr(self): self.check2dMatrix("tlatr")


    def test2DMatrix_tlfla(self): self.check2dMatrix("tlfla")
    def test2DMatrix_tlflf(self): self.check2dMatrix("tlflf")
    def test2DMatrix_tlfll(self): self.check2dMatrix("tlfll")
    def test2DMatrix_tlfls(self): self.check2dMatrix("tlfls")
    def test2DMatrix_tlflr(self): self.check2dMatrix("tlflr")

    def test2DMatrix_tlfta(self): self.check2dMatrix("tlfta")
    def test2DMatrix_tlftf(self): self.check2dMatrix("tlftf")
    def test2DMatrix_tlftl(self): self.check2dMatrix("tlftl")
    def test2DMatrix_tlfts(self): self.check2dMatrix("tlfts")
    def test2DMatrix_tlftr(self): self.check2dMatrix("tlftr")


    def test2DMatrix_tllla(self): self.check2dMatrix("tllla")
    def test2DMatrix_tlllf(self): self.check2dMatrix("tlllf")
    def test2DMatrix_tllll(self): self.check2dMatrix("tllll")
    def test2DMatrix_tllls(self): self.check2dMatrix("tllls")
    def test2DMatrix_tlllr(self): self.check2dMatrix("tlllr")

    def test2DMatrix_tllta(self): self.check2dMatrix("tllta")
    def test2DMatrix_tlltf(self): self.check2dMatrix("tlltf")
    def test2DMatrix_tlltl(self): self.check2dMatrix("tlltl")
    def test2DMatrix_tllts(self): self.check2dMatrix("tllts")
    def test2DMatrix_tlltr(self): self.check2dMatrix("tlltr")


    def test2DMatrix_tlsla(self): self.check2dMatrix("tlsla")
    def test2DMatrix_tlslf(self): self.check2dMatrix("tlslf")
    def test2DMatrix_tlsll(self): self.check2dMatrix("tlsll")
    def test2DMatrix_tlsls(self): self.check2dMatrix("tlsls")
    def test2DMatrix_tlslr(self): self.check2dMatrix("tlslr")

    def test2DMatrix_tlsta(self): self.check2dMatrix("tlsta")
    def test2DMatrix_tlstf(self): self.check2dMatrix("tlstf")
    def test2DMatrix_tlstl(self): self.check2dMatrix("tlstl")
    def test2DMatrix_tlsts(self): self.check2dMatrix("tlsts")
    def test2DMatrix_tlstr(self): self.check2dMatrix("tlstr")


    def test2DMatrix_tlrla(self): self.check2dMatrix("tlrla")
    def test2DMatrix_tlrlf(self): self.check2dMatrix("tlrlf")
    def test2DMatrix_tlrll(self): self.check2dMatrix("tlrll")
    def test2DMatrix_tlrls(self): self.check2dMatrix("tlrls")
    def test2DMatrix_tlrlr(self): self.check2dMatrix("tlrlr")

    def test2DMatrix_tlrta(self): self.check2dMatrix("tlrta")
    def test2DMatrix_tlrtf(self): self.check2dMatrix("tlrtf")
    def test2DMatrix_tlrtl(self): self.check2dMatrix("tlrtl")
    def test2DMatrix_tlrts(self): self.check2dMatrix("tlrts")
    def test2DMatrix_tlrtr(self): self.check2dMatrix("tlrtr")

class Test2dMatrixNonSquare(unittest.TestCase):
    def check2dMatrixNonSquare(self, opts):

        values = {}

        indices = {}
        indices["t"] = (0,3)
        indices["n"] = "a"
        indices["N"] = "a"
        indices["l"] = [0,1,2]
        indices["a"] = ar([0,1,2],dtype=uint)
        indices["r"] = ar([0,0,1,1,2,2],dtype=uint)[::2]
        indices["f"] = ar([0,1,2],dtype=float64)
        indices["e"] = None  # empty

        A = [[1,0,  0],
             [0,0,  0.5],
             [0,0.5,0],
	     [1, 0, 0]]

        values = {}
        values["L"] = A
        values["l"] = [ar(le) for le in A]
        values["a"] = ar(A)

        targets = {}
        targets["s"] = 1
        targets["l"] = [1,1,1,1]
        targets["a"] = ar([1,1,1,1],dtype=uint)
        targets["r"] = ar([1,1,1,1,1,1,1,1],dtype=uint)[::2]
        targets["f"] = ar([1,1,1,1],dtype=float64)

        targets_t = {}
        targets_t["t"] = tuple
        targets_t["l"] = list

        targets_u = {}
        targets_u["s"] = 10
        targets_u["l"] = [10,10,10,10]
        targets_u["a"] = ar([10,10,10,10],dtype=uint)
        targets_u["r"] = ar([10,10,10,10,10,10,10,10],dtype=uint)[::2]
        targets_u["f"] = ar([10,10,10,10],dtype=float64)

        lp = LP()

        if opts[0] == "N":
            lp.getIndexBlock(indices["N"], 3)

        io = indices[opts[0]]
        vl = values [opts[1]]
        
        if len(opts) == 3:
            tr = targets[opts[2]]
            cstr = ">="
        else:
            tr = targets_t[opts[3]]([targets[opts[2]], targets[opts[4]]])
            cstr = "in"

        ob = [1,2,3]
        
        c_ret_idx = [0,1,2,3]

        if io is None:
            ret = lp.addConstraint(vl, cstr, tr)
            self.assert_(ret == c_ret_idx, "%s != %s" %(str(ret), str(c_ret_idx)))
            lp.setObjective(ob)
        else:
            ret = lp.addConstraint( (io, vl), cstr, tr)
            self.assert_(ret == c_ret_idx, "%s != %s" %(str(ret), str(c_ret_idx)))
            lp.setObjective( (io, ob) )

        for num_times in range(2):  # make sure it's same anser second time
            lp.solve()

            self.assertAlmostEqual(lp.getObjectiveValue(), 11)

            v = lp.getSolution()

            self.assert_(len(v) == 3)
            self.assertAlmostEqual(v[0], 1)
            self.assertAlmostEqual(v[1], 2)
            self.assertAlmostEqual(v[2], 2)


    def test2DMatrixNonSquare_tLs(self): self.check2dMatrixNonSquare("tLs")
    def test2DMatrixNonSquare_tLl(self): self.check2dMatrixNonSquare("tLl")
    def test2DMatrixNonSquare_tLa(self): self.check2dMatrixNonSquare("tLa")
    def test2DMatrixNonSquare_tLf(self): self.check2dMatrixNonSquare("tLf")
    def test2DMatrixNonSquare_tLr(self): self.check2dMatrixNonSquare("tLr")

    def test2DMatrixNonSquare_tls(self): self.check2dMatrixNonSquare("tls")
    def test2DMatrixNonSquare_tll(self): self.check2dMatrixNonSquare("tll")
    def test2DMatrixNonSquare_tla(self): self.check2dMatrixNonSquare("tla")
    def test2DMatrixNonSquare_tlf(self): self.check2dMatrixNonSquare("tlf")
    def test2DMatrixNonSquare_tlr(self): self.check2dMatrixNonSquare("tlr")

    def test2DMatrixNonSquare_tas(self): self.check2dMatrixNonSquare("tas")
    def test2DMatrixNonSquare_tal(self): self.check2dMatrixNonSquare("tal")
    def test2DMatrixNonSquare_taa(self): self.check2dMatrixNonSquare("taa")
    def test2DMatrixNonSquare_taf(self): self.check2dMatrixNonSquare("taf")
    def test2DMatrixNonSquare_tar(self): self.check2dMatrixNonSquare("tar")

    
    def test2DMatrixNonSquare_nLs(self): self.check2dMatrixNonSquare("nLs")
    def test2DMatrixNonSquare_nLl(self): self.check2dMatrixNonSquare("nLl")
    def test2DMatrixNonSquare_nLa(self): self.check2dMatrixNonSquare("nLa")
    def test2DMatrixNonSquare_nLf(self): self.check2dMatrixNonSquare("nLf")
    def test2DMatrixNonSquare_nLr(self): self.check2dMatrixNonSquare("nLr")

    def test2DMatrixNonSquare_nls(self): self.check2dMatrixNonSquare("nls")
    def test2DMatrixNonSquare_nll(self): self.check2dMatrixNonSquare("nll")
    def test2DMatrixNonSquare_nla(self): self.check2dMatrixNonSquare("nla")
    def test2DMatrixNonSquare_nlf(self): self.check2dMatrixNonSquare("nlf")
    def test2DMatrixNonSquare_nlr(self): self.check2dMatrixNonSquare("nlr")

    def test2DMatrixNonSquare_nas(self): self.check2dMatrixNonSquare("nas")
    def test2DMatrixNonSquare_nal(self): self.check2dMatrixNonSquare("nal")
    def test2DMatrixNonSquare_naa(self): self.check2dMatrixNonSquare("naa")
    def test2DMatrixNonSquare_naf(self): self.check2dMatrixNonSquare("naf")


    def test2DMatrixNonSquare_NLs(self): self.check2dMatrixNonSquare("NLs")
    def test2DMatrixNonSquare_NLl(self): self.check2dMatrixNonSquare("NLl")
    def test2DMatrixNonSquare_NLa(self): self.check2dMatrixNonSquare("NLa")
    def test2DMatrixNonSquare_NLf(self): self.check2dMatrixNonSquare("NLf")
    def test2DMatrixNonSquare_NLr(self): self.check2dMatrixNonSquare("NLr")

    def test2DMatrixNonSquare_Nls(self): self.check2dMatrixNonSquare("Nls")
    def test2DMatrixNonSquare_Nll(self): self.check2dMatrixNonSquare("Nll")
    def test2DMatrixNonSquare_Nla(self): self.check2dMatrixNonSquare("Nla")
    def test2DMatrixNonSquare_Nlf(self): self.check2dMatrixNonSquare("Nlf")
    def test2DMatrixNonSquare_Nlr(self): self.check2dMatrixNonSquare("Nlr")

    def test2DMatrixNonSquare_Nas(self): self.check2dMatrixNonSquare("Nas")
    def test2DMatrixNonSquare_Nal(self): self.check2dMatrixNonSquare("Nal")
    def test2DMatrixNonSquare_Naa(self): self.check2dMatrixNonSquare("Naa")
    def test2DMatrixNonSquare_Naf(self): self.check2dMatrixNonSquare("Naf")
    def test2DMatrixNonSquare_Nar(self): self.check2dMatrixNonSquare("Nar")


    def test2DMatrixNonSquare_lLs(self): self.check2dMatrixNonSquare("lLs")
    def test2DMatrixNonSquare_lLl(self): self.check2dMatrixNonSquare("lLl")
    def test2DMatrixNonSquare_lLa(self): self.check2dMatrixNonSquare("lLa")
    def test2DMatrixNonSquare_lLf(self): self.check2dMatrixNonSquare("lLf")
    def test2DMatrixNonSquare_lLr(self): self.check2dMatrixNonSquare("lLr")

    def test2DMatrixNonSquare_lls(self): self.check2dMatrixNonSquare("lls")
    def test2DMatrixNonSquare_lll(self): self.check2dMatrixNonSquare("lll")
    def test2DMatrixNonSquare_lla(self): self.check2dMatrixNonSquare("lla")
    def test2DMatrixNonSquare_llf(self): self.check2dMatrixNonSquare("llf")
    def test2DMatrixNonSquare_llr(self): self.check2dMatrixNonSquare("llr")

    def test2DMatrixNonSquare_las(self): self.check2dMatrixNonSquare("las")
    def test2DMatrixNonSquare_lal(self): self.check2dMatrixNonSquare("lal")
    def test2DMatrixNonSquare_laa(self): self.check2dMatrixNonSquare("laa")
    def test2DMatrixNonSquare_laf(self): self.check2dMatrixNonSquare("laf")
    def test2DMatrixNonSquare_lar(self): self.check2dMatrixNonSquare("lar")


    def test2DMatrixNonSquare_aLs(self): self.check2dMatrixNonSquare("aLs")
    def test2DMatrixNonSquare_aLl(self): self.check2dMatrixNonSquare("aLl")
    def test2DMatrixNonSquare_aLa(self): self.check2dMatrixNonSquare("aLa")
    def test2DMatrixNonSquare_aLf(self): self.check2dMatrixNonSquare("aLf")
    def test2DMatrixNonSquare_aLr(self): self.check2dMatrixNonSquare("aLr")

    def test2DMatrixNonSquare_als(self): self.check2dMatrixNonSquare("als")
    def test2DMatrixNonSquare_all(self): self.check2dMatrixNonSquare("all")
    def test2DMatrixNonSquare_ala(self): self.check2dMatrixNonSquare("ala")
    def test2DMatrixNonSquare_alf(self): self.check2dMatrixNonSquare("alf")
    def test2DMatrixNonSquare_alr(self): self.check2dMatrixNonSquare("alr")

    def test2DMatrixNonSquare_aas(self): self.check2dMatrixNonSquare("aas")
    def test2DMatrixNonSquare_aal(self): self.check2dMatrixNonSquare("aal")
    def test2DMatrixNonSquare_aaa(self): self.check2dMatrixNonSquare("aaa")
    def test2DMatrixNonSquare_aaf(self): self.check2dMatrixNonSquare("aaf")
    def test2DMatrixNonSquare_aar(self): self.check2dMatrixNonSquare("aar")


    def test2DMatrixNonSquare_fLs(self): self.check2dMatrixNonSquare("fLs")
    def test2DMatrixNonSquare_fLl(self): self.check2dMatrixNonSquare("fLl")
    def test2DMatrixNonSquare_fLa(self): self.check2dMatrixNonSquare("fLa")
    def test2DMatrixNonSquare_fLf(self): self.check2dMatrixNonSquare("fLf")
    def test2DMatrixNonSquare_fLr(self): self.check2dMatrixNonSquare("fLr")

    def test2DMatrixNonSquare_fls(self): self.check2dMatrixNonSquare("fls")
    def test2DMatrixNonSquare_fll(self): self.check2dMatrixNonSquare("fll")
    def test2DMatrixNonSquare_fla(self): self.check2dMatrixNonSquare("fla")
    def test2DMatrixNonSquare_flf(self): self.check2dMatrixNonSquare("flf")
    def test2DMatrixNonSquare_flr(self): self.check2dMatrixNonSquare("flr")

    def test2DMatrixNonSquare_fas(self): self.check2dMatrixNonSquare("fas")
    def test2DMatrixNonSquare_fal(self): self.check2dMatrixNonSquare("fal")
    def test2DMatrixNonSquare_faa(self): self.check2dMatrixNonSquare("faa")
    def test2DMatrixNonSquare_faf(self): self.check2dMatrixNonSquare("faf")
    def test2DMatrixNonSquare_far(self): self.check2dMatrixNonSquare("far")


    def test2DMatrixNonSquare_eLs(self): self.check2dMatrixNonSquare("eLs")
    def test2DMatrixNonSquare_eLl(self): self.check2dMatrixNonSquare("eLl")
    def test2DMatrixNonSquare_eLa(self): self.check2dMatrixNonSquare("eLa")
    def test2DMatrixNonSquare_eLf(self): self.check2dMatrixNonSquare("eLf")
    def test2DMatrixNonSquare_eLr(self): self.check2dMatrixNonSquare("eLr")

    def test2DMatrixNonSquare_els(self): self.check2dMatrixNonSquare("els")
    def test2DMatrixNonSquare_ell(self): self.check2dMatrixNonSquare("ell")
    def test2DMatrixNonSquare_ela(self): self.check2dMatrixNonSquare("ela")
    def test2DMatrixNonSquare_elf(self): self.check2dMatrixNonSquare("elf")
    def test2DMatrixNonSquare_elr(self): self.check2dMatrixNonSquare("elr")

    def test2DMatrixNonSquare_eas(self): self.check2dMatrixNonSquare("eas")
    def test2DMatrixNonSquare_eal(self): self.check2dMatrixNonSquare("eal")
    def test2DMatrixNonSquare_eaa(self): self.check2dMatrixNonSquare("eaa")
    def test2DMatrixNonSquare_eaf(self): self.check2dMatrixNonSquare("eaf")
    def test2DMatrixNonSquare_ear(self): self.check2dMatrixNonSquare("ear")


    def test2DMatrixNonSquare_rLs(self): self.check2dMatrixNonSquare("rLs")
    def test2DMatrixNonSquare_rLl(self): self.check2dMatrixNonSquare("rLl")
    def test2DMatrixNonSquare_rLa(self): self.check2dMatrixNonSquare("rLa")
    def test2DMatrixNonSquare_rLf(self): self.check2dMatrixNonSquare("rLf")
    def test2DMatrixNonSquare_rLr(self): self.check2dMatrixNonSquare("rLr")

    def test2DMatrixNonSquare_rls(self): self.check2dMatrixNonSquare("rls")
    def test2DMatrixNonSquare_rll(self): self.check2dMatrixNonSquare("rll")
    def test2DMatrixNonSquare_rla(self): self.check2dMatrixNonSquare("rla")
    def test2DMatrixNonSquare_rlf(self): self.check2dMatrixNonSquare("rlf")
    def test2DMatrixNonSquare_rlr(self): self.check2dMatrixNonSquare("rlr")

    def test2DMatrixNonSquare_ras(self): self.check2dMatrixNonSquare("ras")
    def test2DMatrixNonSquare_ral(self): self.check2dMatrixNonSquare("ral")
    def test2DMatrixNonSquare_raa(self): self.check2dMatrixNonSquare("raa")
    def test2DMatrixNonSquare_raf(self): self.check2dMatrixNonSquare("raf")
    def test2DMatrixNonSquare_rar(self): self.check2dMatrixNonSquare("rar")


    # Test a restricted set to look at the bounds; namely fix the tuple at the beginning
    def test2DMatrixNonSquare_tLala(self): self.check2dMatrixNonSquare("tLala")
    def test2DMatrixNonSquare_tLalf(self): self.check2dMatrixNonSquare("tLalf")
    def test2DMatrixNonSquare_tLall(self): self.check2dMatrixNonSquare("tLall")
    def test2DMatrixNonSquare_tLals(self): self.check2dMatrixNonSquare("tLals")
    def test2DMatrixNonSquare_tLalr(self): self.check2dMatrixNonSquare("tLalr")

    def test2DMatrixNonSquare_tLata(self): self.check2dMatrixNonSquare("tLata")
    def test2DMatrixNonSquare_tLatf(self): self.check2dMatrixNonSquare("tLatf")
    def test2DMatrixNonSquare_tLatl(self): self.check2dMatrixNonSquare("tLatl")
    def test2DMatrixNonSquare_tLats(self): self.check2dMatrixNonSquare("tLats")
    def test2DMatrixNonSquare_tLatr(self): self.check2dMatrixNonSquare("tLatr")


    def test2DMatrixNonSquare_tLfla(self): self.check2dMatrixNonSquare("tLfla")
    def test2DMatrixNonSquare_tLflf(self): self.check2dMatrixNonSquare("tLflf")
    def test2DMatrixNonSquare_tLfll(self): self.check2dMatrixNonSquare("tLfll")
    def test2DMatrixNonSquare_tLfls(self): self.check2dMatrixNonSquare("tLfls")
    def test2DMatrixNonSquare_tLflr(self): self.check2dMatrixNonSquare("tLflr")

    def test2DMatrixNonSquare_tLfta(self): self.check2dMatrixNonSquare("tLfta")
    def test2DMatrixNonSquare_tLftf(self): self.check2dMatrixNonSquare("tLftf")
    def test2DMatrixNonSquare_tLftl(self): self.check2dMatrixNonSquare("tLftl")
    def test2DMatrixNonSquare_tLfts(self): self.check2dMatrixNonSquare("tLfts")
    def test2DMatrixNonSquare_tLftr(self): self.check2dMatrixNonSquare("tLftr")


    def test2DMatrixNonSquare_tLlla(self): self.check2dMatrixNonSquare("tLlla")
    def test2DMatrixNonSquare_tLllf(self): self.check2dMatrixNonSquare("tLllf")
    def test2DMatrixNonSquare_tLlll(self): self.check2dMatrixNonSquare("tLlll")
    def test2DMatrixNonSquare_tLlls(self): self.check2dMatrixNonSquare("tLlls")
    def test2DMatrixNonSquare_tLllr(self): self.check2dMatrixNonSquare("tLllr")

    def test2DMatrixNonSquare_tLlta(self): self.check2dMatrixNonSquare("tLlta")
    def test2DMatrixNonSquare_tLltf(self): self.check2dMatrixNonSquare("tLltf")
    def test2DMatrixNonSquare_tLltl(self): self.check2dMatrixNonSquare("tLltl")
    def test2DMatrixNonSquare_tLlts(self): self.check2dMatrixNonSquare("tLlts")
    def test2DMatrixNonSquare_tLltr(self): self.check2dMatrixNonSquare("tLltr")


    def test2DMatrixNonSquare_tLsla(self): self.check2dMatrixNonSquare("tLsla")
    def test2DMatrixNonSquare_tLslf(self): self.check2dMatrixNonSquare("tLslf")
    def test2DMatrixNonSquare_tLsll(self): self.check2dMatrixNonSquare("tLsll")
    def test2DMatrixNonSquare_tLsls(self): self.check2dMatrixNonSquare("tLsls")
    def test2DMatrixNonSquare_tLslr(self): self.check2dMatrixNonSquare("tLslr")

    def test2DMatrixNonSquare_tLsta(self): self.check2dMatrixNonSquare("tLsta")
    def test2DMatrixNonSquare_tLstf(self): self.check2dMatrixNonSquare("tLstf")
    def test2DMatrixNonSquare_tLstl(self): self.check2dMatrixNonSquare("tLstl")
    def test2DMatrixNonSquare_tLsts(self): self.check2dMatrixNonSquare("tLsts")
    def test2DMatrixNonSquare_tLstr(self): self.check2dMatrixNonSquare("tLstr")


    def test2DMatrixNonSquare_tLrla(self): self.check2dMatrixNonSquare("tLrla")
    def test2DMatrixNonSquare_tLrlf(self): self.check2dMatrixNonSquare("tLrlf")
    def test2DMatrixNonSquare_tLrll(self): self.check2dMatrixNonSquare("tLrll")
    def test2DMatrixNonSquare_tLrls(self): self.check2dMatrixNonSquare("tLrls")
    def test2DMatrixNonSquare_tLrlr(self): self.check2dMatrixNonSquare("tLrlr")

    def test2DMatrixNonSquare_tLrta(self): self.check2dMatrixNonSquare("tLrta")
    def test2DMatrixNonSquare_tLrtf(self): self.check2dMatrixNonSquare("tLrtf")
    def test2DMatrixNonSquare_tLrtl(self): self.check2dMatrixNonSquare("tLrtl")
    def test2DMatrixNonSquare_tLrts(self): self.check2dMatrixNonSquare("tLrts")
    def test2DMatrixNonSquare_tLrtr(self): self.check2dMatrixNonSquare("tLrtr")


    def test2DMatrixNonSquare_taala(self): self.check2dMatrixNonSquare("taala")
    def test2DMatrixNonSquare_taalf(self): self.check2dMatrixNonSquare("taalf")
    def test2DMatrixNonSquare_taall(self): self.check2dMatrixNonSquare("taall")
    def test2DMatrixNonSquare_taals(self): self.check2dMatrixNonSquare("taals")
    def test2DMatrixNonSquare_taalr(self): self.check2dMatrixNonSquare("taalr")

    def test2DMatrixNonSquare_taata(self): self.check2dMatrixNonSquare("taata")
    def test2DMatrixNonSquare_taatf(self): self.check2dMatrixNonSquare("taatf")
    def test2DMatrixNonSquare_taatl(self): self.check2dMatrixNonSquare("taatl")
    def test2DMatrixNonSquare_taats(self): self.check2dMatrixNonSquare("taats")
    def test2DMatrixNonSquare_taatr(self): self.check2dMatrixNonSquare("taatr")


    def test2DMatrixNonSquare_tafla(self): self.check2dMatrixNonSquare("tafla")
    def test2DMatrixNonSquare_taflf(self): self.check2dMatrixNonSquare("taflf")
    def test2DMatrixNonSquare_tafll(self): self.check2dMatrixNonSquare("tafll")
    def test2DMatrixNonSquare_tafls(self): self.check2dMatrixNonSquare("tafls")
    def test2DMatrixNonSquare_taflr(self): self.check2dMatrixNonSquare("taflr")

    def test2DMatrixNonSquare_tafta(self): self.check2dMatrixNonSquare("tafta")
    def test2DMatrixNonSquare_taftf(self): self.check2dMatrixNonSquare("taftf")
    def test2DMatrixNonSquare_taftl(self): self.check2dMatrixNonSquare("taftl")
    def test2DMatrixNonSquare_tafts(self): self.check2dMatrixNonSquare("tafts")
    def test2DMatrixNonSquare_taftr(self): self.check2dMatrixNonSquare("taftr")


    def test2DMatrixNonSquare_talla(self): self.check2dMatrixNonSquare("talla")
    def test2DMatrixNonSquare_tallf(self): self.check2dMatrixNonSquare("tallf")
    def test2DMatrixNonSquare_talll(self): self.check2dMatrixNonSquare("talll")
    def test2DMatrixNonSquare_talls(self): self.check2dMatrixNonSquare("talls")
    def test2DMatrixNonSquare_tallr(self): self.check2dMatrixNonSquare("tallr")

    def test2DMatrixNonSquare_talta(self): self.check2dMatrixNonSquare("talta")
    def test2DMatrixNonSquare_taltf(self): self.check2dMatrixNonSquare("taltf")
    def test2DMatrixNonSquare_taltl(self): self.check2dMatrixNonSquare("taltl")
    def test2DMatrixNonSquare_talts(self): self.check2dMatrixNonSquare("talts")
    def test2DMatrixNonSquare_taltr(self): self.check2dMatrixNonSquare("taltr")


    def test2DMatrixNonSquare_tasla(self): self.check2dMatrixNonSquare("tasla")
    def test2DMatrixNonSquare_taslf(self): self.check2dMatrixNonSquare("taslf")
    def test2DMatrixNonSquare_tasll(self): self.check2dMatrixNonSquare("tasll")
    def test2DMatrixNonSquare_tasls(self): self.check2dMatrixNonSquare("tasls")
    def test2DMatrixNonSquare_taslr(self): self.check2dMatrixNonSquare("taslr")

    def test2DMatrixNonSquare_tasta(self): self.check2dMatrixNonSquare("tasta")
    def test2DMatrixNonSquare_tastf(self): self.check2dMatrixNonSquare("tastf")
    def test2DMatrixNonSquare_tastl(self): self.check2dMatrixNonSquare("tastl")
    def test2DMatrixNonSquare_tasts(self): self.check2dMatrixNonSquare("tasts")
    def test2DMatrixNonSquare_tastr(self): self.check2dMatrixNonSquare("tastr")


    def test2DMatrixNonSquare_tarla(self): self.check2dMatrixNonSquare("tarla")
    def test2DMatrixNonSquare_tarlf(self): self.check2dMatrixNonSquare("tarlf")
    def test2DMatrixNonSquare_tarll(self): self.check2dMatrixNonSquare("tarll")
    def test2DMatrixNonSquare_tarls(self): self.check2dMatrixNonSquare("tarls")
    def test2DMatrixNonSquare_tarlr(self): self.check2dMatrixNonSquare("tarlr")

    def test2DMatrixNonSquare_tarta(self): self.check2dMatrixNonSquare("tarta")
    def test2DMatrixNonSquare_tartf(self): self.check2dMatrixNonSquare("tartf")
    def test2DMatrixNonSquare_tartl(self): self.check2dMatrixNonSquare("tartl")
    def test2DMatrixNonSquare_tarts(self): self.check2dMatrixNonSquare("tarts")
    def test2DMatrixNonSquare_tartr(self): self.check2dMatrixNonSquare("tartr")


    def test2DMatrixNonSquare_tlala(self): self.check2dMatrixNonSquare("tlala")
    def test2DMatrixNonSquare_tlalf(self): self.check2dMatrixNonSquare("tlalf")
    def test2DMatrixNonSquare_tlall(self): self.check2dMatrixNonSquare("tlall")
    def test2DMatrixNonSquare_tlals(self): self.check2dMatrixNonSquare("tlals")
    def test2DMatrixNonSquare_tlalr(self): self.check2dMatrixNonSquare("tlalr")

    def test2DMatrixNonSquare_tlata(self): self.check2dMatrixNonSquare("tlata")
    def test2DMatrixNonSquare_tlatf(self): self.check2dMatrixNonSquare("tlatf")
    def test2DMatrixNonSquare_tlatl(self): self.check2dMatrixNonSquare("tlatl")
    def test2DMatrixNonSquare_tlats(self): self.check2dMatrixNonSquare("tlats")
    def test2DMatrixNonSquare_tlatr(self): self.check2dMatrixNonSquare("tlatr")


    def test2DMatrixNonSquare_tlfla(self): self.check2dMatrixNonSquare("tlfla")
    def test2DMatrixNonSquare_tlflf(self): self.check2dMatrixNonSquare("tlflf")
    def test2DMatrixNonSquare_tlfll(self): self.check2dMatrixNonSquare("tlfll")
    def test2DMatrixNonSquare_tlfls(self): self.check2dMatrixNonSquare("tlfls")
    def test2DMatrixNonSquare_tlflr(self): self.check2dMatrixNonSquare("tlflr")

    def test2DMatrixNonSquare_tlfta(self): self.check2dMatrixNonSquare("tlfta")
    def test2DMatrixNonSquare_tlftf(self): self.check2dMatrixNonSquare("tlftf")
    def test2DMatrixNonSquare_tlftl(self): self.check2dMatrixNonSquare("tlftl")
    def test2DMatrixNonSquare_tlfts(self): self.check2dMatrixNonSquare("tlfts")
    def test2DMatrixNonSquare_tlftr(self): self.check2dMatrixNonSquare("tlftr")


    def test2DMatrixNonSquare_tllla(self): self.check2dMatrixNonSquare("tllla")
    def test2DMatrixNonSquare_tlllf(self): self.check2dMatrixNonSquare("tlllf")
    def test2DMatrixNonSquare_tllll(self): self.check2dMatrixNonSquare("tllll")
    def test2DMatrixNonSquare_tllls(self): self.check2dMatrixNonSquare("tllls")
    def test2DMatrixNonSquare_tlllr(self): self.check2dMatrixNonSquare("tlllr")

    def test2DMatrixNonSquare_tllta(self): self.check2dMatrixNonSquare("tllta")
    def test2DMatrixNonSquare_tlltf(self): self.check2dMatrixNonSquare("tlltf")
    def test2DMatrixNonSquare_tlltl(self): self.check2dMatrixNonSquare("tlltl")
    def test2DMatrixNonSquare_tllts(self): self.check2dMatrixNonSquare("tllts")
    def test2DMatrixNonSquare_tlltr(self): self.check2dMatrixNonSquare("tlltr")


    def test2DMatrixNonSquare_tlsla(self): self.check2dMatrixNonSquare("tlsla")
    def test2DMatrixNonSquare_tlslf(self): self.check2dMatrixNonSquare("tlslf")
    def test2DMatrixNonSquare_tlsll(self): self.check2dMatrixNonSquare("tlsll")
    def test2DMatrixNonSquare_tlsls(self): self.check2dMatrixNonSquare("tlsls")
    def test2DMatrixNonSquare_tlslr(self): self.check2dMatrixNonSquare("tlslr")

    def test2DMatrixNonSquare_tlsta(self): self.check2dMatrixNonSquare("tlsta")
    def test2DMatrixNonSquare_tlstf(self): self.check2dMatrixNonSquare("tlstf")
    def test2DMatrixNonSquare_tlstl(self): self.check2dMatrixNonSquare("tlstl")
    def test2DMatrixNonSquare_tlsts(self): self.check2dMatrixNonSquare("tlsts")
    def test2DMatrixNonSquare_tlstr(self): self.check2dMatrixNonSquare("tlstr")


    def test2DMatrixNonSquare_tlrla(self): self.check2dMatrixNonSquare("tlrla")
    def test2DMatrixNonSquare_tlrlf(self): self.check2dMatrixNonSquare("tlrlf")
    def test2DMatrixNonSquare_tlrll(self): self.check2dMatrixNonSquare("tlrll")
    def test2DMatrixNonSquare_tlrls(self): self.check2dMatrixNonSquare("tlrls")
    def test2DMatrixNonSquare_tlrlr(self): self.check2dMatrixNonSquare("tlrlr")

    def test2DMatrixNonSquare_tlrta(self): self.check2dMatrixNonSquare("tlrta")
    def test2DMatrixNonSquare_tlrtf(self): self.check2dMatrixNonSquare("tlrtf")
    def test2DMatrixNonSquare_tlrtl(self): self.check2dMatrixNonSquare("tlrtl")
    def test2DMatrixNonSquare_tlrts(self): self.check2dMatrixNonSquare("tlrts")
    def test2DMatrixNonSquare_tlrtr(self): self.check2dMatrixNonSquare("tlrtr")

class Test2dMatrixBlocks(unittest.TestCase):
    def check2dMatrixBlocks(self, opts):

	mtypes = {}
	mtypes["d"] = dict
	mtypes["l"] = list

        values = {}

        idx_1 = {}
        idx_1["t"] = (0,3)
        idx_1["n"] = "a"
        idx_1["N"] = "a"
        idx_1["l"] = [0,1,2]
        idx_1["a"] = ar([0,1,2],dtype=uint)

        idx_2 = {}
        idx_2["t"] = (3,6)
        idx_2["n"] = "b"
        idx_2["N"] = "b"
        idx_2["l"] = [3,4,5]

	values_1 = {}
	values_1["s"] = 10
	values_1["m"] = 10*ones( (4,3) )
	values_1["a"] = 10*ones( 3 )
	values_1["l"] = [10, 10, 10]
	values_1["L"] = [[10, 10, 10],[10, 10, 10],[10, 10, 10],[10, 10, 10]]
	values_1["A"] = [ar([10, 10, 10]),ar([10, 10, 10]),[10, 10, 10],[10, 10, 10]]

	A2 = [[0, 1, 0], [0, 0, 2], [1,0,0], [0,0,1]]

	values_2 = {}
	values_2["L"] = A2
	values_2["A"] = ar(A2)

	constraint_rhs = {}
	constraint_rhs["l"] = (["<=", 10], [">=", 0])
	constraint_rhs["L"] = (["<=", [10,10,10,10]], [">=", 0])
	constraint_rhs["B"] = ["in", [0, 10]]
	constraint_rhs["A"] = ["in", ([0,0,0,0] , 10)]
	

	solution = ar([0,0,0, 10, 10, 5])

        lp = LP()

        if opts[1] == "N":
            lp.getIndexBlock(idx_1["N"], 3)

        if opts[2] == "N":
            lp.getIndexBlock(idx_2["N"], 3)

	mtype= mtypes[opts[0]]
        id_1 = idx_1[opts[1]]
        id_2 = idx_2[opts[2]]
	v_1  = values_1[opts[3]]
	v_2  = values_2[opts[4]]
	c_rhs= constraint_rhs[opts[5]]

	# print "id_1 = ", id_1
	# print "id_2 = ", id_2
	# print "v_1 = ", v_1
	# print "v_2 = ", v_2

	def addConstraintBlock(idx_block, c, r):
	    ret_idx = lp.addConstraint(mtype([(id_1, v_1), (id_2, v_2)]), c, r)
	    self.assert_(ret_idx == idx_block, "%s != %s (true)" %(str(ret_idx), str(idx_block)))

	if type(c_rhs) is tuple:
	    cr1, cr2 = c_rhs
	    addConstraintBlock([0,1,2,3], *cr1)
	    addConstraintBlock([4,5,6,7], *cr2)
	else:
	    addConstraintBlock([0,1,2,3], *c_rhs)


	lp.setObjective([1]*6, mode = "maximize")

        for num_times in range(2):  # make sure it's same answer second time solving it
            lp.solve()

            self.assertAlmostEqual(lp.getObjectiveValue(), solution.sum())

            v = lp.getSolution()

	    self.assert_(len(v) == len(solution))

	    for i, s in enumerate(solution):
		self.assertAlmostEqual(v[i], s)


    def test2dMatrixBlocks_dttsLl(self): self.check2dMatrixBlocks("dttsLl")
    def test2dMatrixBlocks_dttsLL(self): self.check2dMatrixBlocks("dttsLL")
    def test2dMatrixBlocks_dttsLB(self): self.check2dMatrixBlocks("dttsLB")
    def test2dMatrixBlocks_dttsLA(self): self.check2dMatrixBlocks("dttsLA")

    def test2dMatrixBlocks_dttsAl(self): self.check2dMatrixBlocks("dttsAl")
    def test2dMatrixBlocks_dttsAL(self): self.check2dMatrixBlocks("dttsAL")
    def test2dMatrixBlocks_dttsAB(self): self.check2dMatrixBlocks("dttsAB")
    def test2dMatrixBlocks_dttsAA(self): self.check2dMatrixBlocks("dttsAA")

    def test2dMatrixBlocks_dttmLl(self): self.check2dMatrixBlocks("dttmLl")
    def test2dMatrixBlocks_dttmLL(self): self.check2dMatrixBlocks("dttmLL")
    def test2dMatrixBlocks_dttmLB(self): self.check2dMatrixBlocks("dttmLB")
    def test2dMatrixBlocks_dttmLA(self): self.check2dMatrixBlocks("dttmLA")

    def test2dMatrixBlocks_dttmAl(self): self.check2dMatrixBlocks("dttmAl")
    def test2dMatrixBlocks_dttmAL(self): self.check2dMatrixBlocks("dttmAL")
    def test2dMatrixBlocks_dttmAB(self): self.check2dMatrixBlocks("dttmAB")
    def test2dMatrixBlocks_dttmAA(self): self.check2dMatrixBlocks("dttmAA")

    def test2dMatrixBlocks_dttaLl(self): self.check2dMatrixBlocks("dttaLl")
    def test2dMatrixBlocks_dttaLL(self): self.check2dMatrixBlocks("dttaLL")
    def test2dMatrixBlocks_dttaLB(self): self.check2dMatrixBlocks("dttaLB")
    def test2dMatrixBlocks_dttaLA(self): self.check2dMatrixBlocks("dttaLA")

    def test2dMatrixBlocks_dttaAl(self): self.check2dMatrixBlocks("dttaAl")
    def test2dMatrixBlocks_dttaAL(self): self.check2dMatrixBlocks("dttaAL")
    def test2dMatrixBlocks_dttaAB(self): self.check2dMatrixBlocks("dttaAB")
    def test2dMatrixBlocks_dttaAA(self): self.check2dMatrixBlocks("dttaAA")

    def test2dMatrixBlocks_dttlLl(self): self.check2dMatrixBlocks("dttlLl")
    def test2dMatrixBlocks_dttlLL(self): self.check2dMatrixBlocks("dttlLL")
    def test2dMatrixBlocks_dttlLB(self): self.check2dMatrixBlocks("dttlLB")
    def test2dMatrixBlocks_dttlLA(self): self.check2dMatrixBlocks("dttlLA")

    def test2dMatrixBlocks_dttlAl(self): self.check2dMatrixBlocks("dttlAl")
    def test2dMatrixBlocks_dttlAL(self): self.check2dMatrixBlocks("dttlAL")
    def test2dMatrixBlocks_dttlAB(self): self.check2dMatrixBlocks("dttlAB")
    def test2dMatrixBlocks_dttlAA(self): self.check2dMatrixBlocks("dttlAA")

    def test2dMatrixBlocks_dttLLl(self): self.check2dMatrixBlocks("dttLLl")
    def test2dMatrixBlocks_dttLLL(self): self.check2dMatrixBlocks("dttLLL")
    def test2dMatrixBlocks_dttLLB(self): self.check2dMatrixBlocks("dttLLB")
    def test2dMatrixBlocks_dttLLA(self): self.check2dMatrixBlocks("dttLLA")

    def test2dMatrixBlocks_dttLAl(self): self.check2dMatrixBlocks("dttLAl")
    def test2dMatrixBlocks_dttLAL(self): self.check2dMatrixBlocks("dttLAL")
    def test2dMatrixBlocks_dttLAB(self): self.check2dMatrixBlocks("dttLAB")
    def test2dMatrixBlocks_dttLAA(self): self.check2dMatrixBlocks("dttLAA")

    def test2dMatrixBlocks_dttALl(self): self.check2dMatrixBlocks("dttALl")
    def test2dMatrixBlocks_dttALL(self): self.check2dMatrixBlocks("dttALL")
    def test2dMatrixBlocks_dttALB(self): self.check2dMatrixBlocks("dttALB")
    def test2dMatrixBlocks_dttALA(self): self.check2dMatrixBlocks("dttALA")

    def test2dMatrixBlocks_dttAAl(self): self.check2dMatrixBlocks("dttAAl")
    def test2dMatrixBlocks_dttAAL(self): self.check2dMatrixBlocks("dttAAL")
    def test2dMatrixBlocks_dttAAB(self): self.check2dMatrixBlocks("dttAAB")
    def test2dMatrixBlocks_dttAAA(self): self.check2dMatrixBlocks("dttAAA")

    def test2dMatrixBlocks_dtnsLl(self): self.check2dMatrixBlocks("dtnsLl")
    def test2dMatrixBlocks_dtnsLL(self): self.check2dMatrixBlocks("dtnsLL")
    def test2dMatrixBlocks_dtnsLB(self): self.check2dMatrixBlocks("dtnsLB")
    def test2dMatrixBlocks_dtnsLA(self): self.check2dMatrixBlocks("dtnsLA")

    def test2dMatrixBlocks_dtnsAl(self): self.check2dMatrixBlocks("dtnsAl")
    def test2dMatrixBlocks_dtnsAL(self): self.check2dMatrixBlocks("dtnsAL")
    def test2dMatrixBlocks_dtnsAB(self): self.check2dMatrixBlocks("dtnsAB")
    def test2dMatrixBlocks_dtnsAA(self): self.check2dMatrixBlocks("dtnsAA")

    def test2dMatrixBlocks_dtnmLl(self): self.check2dMatrixBlocks("dtnmLl")
    def test2dMatrixBlocks_dtnmLL(self): self.check2dMatrixBlocks("dtnmLL")
    def test2dMatrixBlocks_dtnmLB(self): self.check2dMatrixBlocks("dtnmLB")
    def test2dMatrixBlocks_dtnmLA(self): self.check2dMatrixBlocks("dtnmLA")

    def test2dMatrixBlocks_dtnmAl(self): self.check2dMatrixBlocks("dtnmAl")
    def test2dMatrixBlocks_dtnmAL(self): self.check2dMatrixBlocks("dtnmAL")
    def test2dMatrixBlocks_dtnmAB(self): self.check2dMatrixBlocks("dtnmAB")
    def test2dMatrixBlocks_dtnmAA(self): self.check2dMatrixBlocks("dtnmAA")

    def test2dMatrixBlocks_dtnaLl(self): self.check2dMatrixBlocks("dtnaLl")
    def test2dMatrixBlocks_dtnaLL(self): self.check2dMatrixBlocks("dtnaLL")
    def test2dMatrixBlocks_dtnaLB(self): self.check2dMatrixBlocks("dtnaLB")
    def test2dMatrixBlocks_dtnaLA(self): self.check2dMatrixBlocks("dtnaLA")

    def test2dMatrixBlocks_dtnaAl(self): self.check2dMatrixBlocks("dtnaAl")
    def test2dMatrixBlocks_dtnaAL(self): self.check2dMatrixBlocks("dtnaAL")
    def test2dMatrixBlocks_dtnaAB(self): self.check2dMatrixBlocks("dtnaAB")
    def test2dMatrixBlocks_dtnaAA(self): self.check2dMatrixBlocks("dtnaAA")

    def test2dMatrixBlocks_dtnlLl(self): self.check2dMatrixBlocks("dtnlLl")
    def test2dMatrixBlocks_dtnlLL(self): self.check2dMatrixBlocks("dtnlLL")
    def test2dMatrixBlocks_dtnlLB(self): self.check2dMatrixBlocks("dtnlLB")
    def test2dMatrixBlocks_dtnlLA(self): self.check2dMatrixBlocks("dtnlLA")

    def test2dMatrixBlocks_dtnlAl(self): self.check2dMatrixBlocks("dtnlAl")
    def test2dMatrixBlocks_dtnlAL(self): self.check2dMatrixBlocks("dtnlAL")
    def test2dMatrixBlocks_dtnlAB(self): self.check2dMatrixBlocks("dtnlAB")
    def test2dMatrixBlocks_dtnlAA(self): self.check2dMatrixBlocks("dtnlAA")

    def test2dMatrixBlocks_dtnLLl(self): self.check2dMatrixBlocks("dtnLLl")
    def test2dMatrixBlocks_dtnLLL(self): self.check2dMatrixBlocks("dtnLLL")
    def test2dMatrixBlocks_dtnLLB(self): self.check2dMatrixBlocks("dtnLLB")
    def test2dMatrixBlocks_dtnLLA(self): self.check2dMatrixBlocks("dtnLLA")

    def test2dMatrixBlocks_dtnLAl(self): self.check2dMatrixBlocks("dtnLAl")
    def test2dMatrixBlocks_dtnLAL(self): self.check2dMatrixBlocks("dtnLAL")
    def test2dMatrixBlocks_dtnLAB(self): self.check2dMatrixBlocks("dtnLAB")
    def test2dMatrixBlocks_dtnLAA(self): self.check2dMatrixBlocks("dtnLAA")

    def test2dMatrixBlocks_dtnALl(self): self.check2dMatrixBlocks("dtnALl")
    def test2dMatrixBlocks_dtnALL(self): self.check2dMatrixBlocks("dtnALL")
    def test2dMatrixBlocks_dtnALB(self): self.check2dMatrixBlocks("dtnALB")
    def test2dMatrixBlocks_dtnALA(self): self.check2dMatrixBlocks("dtnALA")

    def test2dMatrixBlocks_dtnAAl(self): self.check2dMatrixBlocks("dtnAAl")
    def test2dMatrixBlocks_dtnAAL(self): self.check2dMatrixBlocks("dtnAAL")
    def test2dMatrixBlocks_dtnAAB(self): self.check2dMatrixBlocks("dtnAAB")
    def test2dMatrixBlocks_dtnAAA(self): self.check2dMatrixBlocks("dtnAAA")

    def test2dMatrixBlocks_dtNsLl(self): self.check2dMatrixBlocks("dtNsLl")
    def test2dMatrixBlocks_dtNsLL(self): self.check2dMatrixBlocks("dtNsLL")
    def test2dMatrixBlocks_dtNsLB(self): self.check2dMatrixBlocks("dtNsLB")
    def test2dMatrixBlocks_dtNsLA(self): self.check2dMatrixBlocks("dtNsLA")

    def test2dMatrixBlocks_dtNsAl(self): self.check2dMatrixBlocks("dtNsAl")
    def test2dMatrixBlocks_dtNsAL(self): self.check2dMatrixBlocks("dtNsAL")
    def test2dMatrixBlocks_dtNsAB(self): self.check2dMatrixBlocks("dtNsAB")
    def test2dMatrixBlocks_dtNsAA(self): self.check2dMatrixBlocks("dtNsAA")

    def test2dMatrixBlocks_dtNmLl(self): self.check2dMatrixBlocks("dtNmLl")
    def test2dMatrixBlocks_dtNmLL(self): self.check2dMatrixBlocks("dtNmLL")
    def test2dMatrixBlocks_dtNmLB(self): self.check2dMatrixBlocks("dtNmLB")
    def test2dMatrixBlocks_dtNmLA(self): self.check2dMatrixBlocks("dtNmLA")

    def test2dMatrixBlocks_dtNmAl(self): self.check2dMatrixBlocks("dtNmAl")
    def test2dMatrixBlocks_dtNmAL(self): self.check2dMatrixBlocks("dtNmAL")
    def test2dMatrixBlocks_dtNmAB(self): self.check2dMatrixBlocks("dtNmAB")
    def test2dMatrixBlocks_dtNmAA(self): self.check2dMatrixBlocks("dtNmAA")

    def test2dMatrixBlocks_dtNaLl(self): self.check2dMatrixBlocks("dtNaLl")
    def test2dMatrixBlocks_dtNaLL(self): self.check2dMatrixBlocks("dtNaLL")
    def test2dMatrixBlocks_dtNaLB(self): self.check2dMatrixBlocks("dtNaLB")
    def test2dMatrixBlocks_dtNaLA(self): self.check2dMatrixBlocks("dtNaLA")

    def test2dMatrixBlocks_dtNaAl(self): self.check2dMatrixBlocks("dtNaAl")
    def test2dMatrixBlocks_dtNaAL(self): self.check2dMatrixBlocks("dtNaAL")
    def test2dMatrixBlocks_dtNaAB(self): self.check2dMatrixBlocks("dtNaAB")
    def test2dMatrixBlocks_dtNaAA(self): self.check2dMatrixBlocks("dtNaAA")

    def test2dMatrixBlocks_dtNlLl(self): self.check2dMatrixBlocks("dtNlLl")
    def test2dMatrixBlocks_dtNlLL(self): self.check2dMatrixBlocks("dtNlLL")
    def test2dMatrixBlocks_dtNlLB(self): self.check2dMatrixBlocks("dtNlLB")
    def test2dMatrixBlocks_dtNlLA(self): self.check2dMatrixBlocks("dtNlLA")

    def test2dMatrixBlocks_dtNlAl(self): self.check2dMatrixBlocks("dtNlAl")
    def test2dMatrixBlocks_dtNlAL(self): self.check2dMatrixBlocks("dtNlAL")
    def test2dMatrixBlocks_dtNlAB(self): self.check2dMatrixBlocks("dtNlAB")
    def test2dMatrixBlocks_dtNlAA(self): self.check2dMatrixBlocks("dtNlAA")

    def test2dMatrixBlocks_dtNLLl(self): self.check2dMatrixBlocks("dtNLLl")
    def test2dMatrixBlocks_dtNLLL(self): self.check2dMatrixBlocks("dtNLLL")
    def test2dMatrixBlocks_dtNLLB(self): self.check2dMatrixBlocks("dtNLLB")
    def test2dMatrixBlocks_dtNLLA(self): self.check2dMatrixBlocks("dtNLLA")

    def test2dMatrixBlocks_dtNLAl(self): self.check2dMatrixBlocks("dtNLAl")
    def test2dMatrixBlocks_dtNLAL(self): self.check2dMatrixBlocks("dtNLAL")
    def test2dMatrixBlocks_dtNLAB(self): self.check2dMatrixBlocks("dtNLAB")
    def test2dMatrixBlocks_dtNLAA(self): self.check2dMatrixBlocks("dtNLAA")

    def test2dMatrixBlocks_dtNALl(self): self.check2dMatrixBlocks("dtNALl")
    def test2dMatrixBlocks_dtNALL(self): self.check2dMatrixBlocks("dtNALL")
    def test2dMatrixBlocks_dtNALB(self): self.check2dMatrixBlocks("dtNALB")
    def test2dMatrixBlocks_dtNALA(self): self.check2dMatrixBlocks("dtNALA")

    def test2dMatrixBlocks_dtNAAl(self): self.check2dMatrixBlocks("dtNAAl")
    def test2dMatrixBlocks_dtNAAL(self): self.check2dMatrixBlocks("dtNAAL")
    def test2dMatrixBlocks_dtNAAB(self): self.check2dMatrixBlocks("dtNAAB")
    def test2dMatrixBlocks_dtNAAA(self): self.check2dMatrixBlocks("dtNAAA")

    def test2dMatrixBlocks_dntmLl(self): self.check2dMatrixBlocks("dntmLl")
    def test2dMatrixBlocks_dntmLL(self): self.check2dMatrixBlocks("dntmLL")
    def test2dMatrixBlocks_dntmLB(self): self.check2dMatrixBlocks("dntmLB")
    def test2dMatrixBlocks_dntmLA(self): self.check2dMatrixBlocks("dntmLA")

    def test2dMatrixBlocks_dntmAl(self): self.check2dMatrixBlocks("dntmAl")
    def test2dMatrixBlocks_dntmAL(self): self.check2dMatrixBlocks("dntmAL")
    def test2dMatrixBlocks_dntmAB(self): self.check2dMatrixBlocks("dntmAB")
    def test2dMatrixBlocks_dntmAA(self): self.check2dMatrixBlocks("dntmAA")

    def test2dMatrixBlocks_dntaLl(self): self.check2dMatrixBlocks("dntaLl")
    def test2dMatrixBlocks_dntaLL(self): self.check2dMatrixBlocks("dntaLL")
    def test2dMatrixBlocks_dntaLB(self): self.check2dMatrixBlocks("dntaLB")
    def test2dMatrixBlocks_dntaLA(self): self.check2dMatrixBlocks("dntaLA")

    def test2dMatrixBlocks_dntaAl(self): self.check2dMatrixBlocks("dntaAl")
    def test2dMatrixBlocks_dntaAL(self): self.check2dMatrixBlocks("dntaAL")
    def test2dMatrixBlocks_dntaAB(self): self.check2dMatrixBlocks("dntaAB")
    def test2dMatrixBlocks_dntaAA(self): self.check2dMatrixBlocks("dntaAA")

    def test2dMatrixBlocks_dntlLl(self): self.check2dMatrixBlocks("dntlLl")
    def test2dMatrixBlocks_dntlLL(self): self.check2dMatrixBlocks("dntlLL")
    def test2dMatrixBlocks_dntlLB(self): self.check2dMatrixBlocks("dntlLB")
    def test2dMatrixBlocks_dntlLA(self): self.check2dMatrixBlocks("dntlLA")

    def test2dMatrixBlocks_dntlAl(self): self.check2dMatrixBlocks("dntlAl")
    def test2dMatrixBlocks_dntlAL(self): self.check2dMatrixBlocks("dntlAL")
    def test2dMatrixBlocks_dntlAB(self): self.check2dMatrixBlocks("dntlAB")
    def test2dMatrixBlocks_dntlAA(self): self.check2dMatrixBlocks("dntlAA")

    def test2dMatrixBlocks_dntLLl(self): self.check2dMatrixBlocks("dntLLl")
    def test2dMatrixBlocks_dntLLL(self): self.check2dMatrixBlocks("dntLLL")
    def test2dMatrixBlocks_dntLLB(self): self.check2dMatrixBlocks("dntLLB")
    def test2dMatrixBlocks_dntLLA(self): self.check2dMatrixBlocks("dntLLA")

    def test2dMatrixBlocks_dntLAl(self): self.check2dMatrixBlocks("dntLAl")
    def test2dMatrixBlocks_dntLAL(self): self.check2dMatrixBlocks("dntLAL")
    def test2dMatrixBlocks_dntLAB(self): self.check2dMatrixBlocks("dntLAB")
    def test2dMatrixBlocks_dntLAA(self): self.check2dMatrixBlocks("dntLAA")

    def test2dMatrixBlocks_dntALl(self): self.check2dMatrixBlocks("dntALl")
    def test2dMatrixBlocks_dntALL(self): self.check2dMatrixBlocks("dntALL")
    def test2dMatrixBlocks_dntALB(self): self.check2dMatrixBlocks("dntALB")
    def test2dMatrixBlocks_dntALA(self): self.check2dMatrixBlocks("dntALA")

    def test2dMatrixBlocks_dntAAl(self): self.check2dMatrixBlocks("dntAAl")
    def test2dMatrixBlocks_dntAAL(self): self.check2dMatrixBlocks("dntAAL")
    def test2dMatrixBlocks_dntAAB(self): self.check2dMatrixBlocks("dntAAB")
    def test2dMatrixBlocks_dntAAA(self): self.check2dMatrixBlocks("dntAAA")

    def test2dMatrixBlocks_dnnmLl(self): self.check2dMatrixBlocks("dnnmLl")
    def test2dMatrixBlocks_dnnmLL(self): self.check2dMatrixBlocks("dnnmLL")
    def test2dMatrixBlocks_dnnmLB(self): self.check2dMatrixBlocks("dnnmLB")
    def test2dMatrixBlocks_dnnmLA(self): self.check2dMatrixBlocks("dnnmLA")

    def test2dMatrixBlocks_dnnmAl(self): self.check2dMatrixBlocks("dnnmAl")
    def test2dMatrixBlocks_dnnmAL(self): self.check2dMatrixBlocks("dnnmAL")
    def test2dMatrixBlocks_dnnmAB(self): self.check2dMatrixBlocks("dnnmAB")
    def test2dMatrixBlocks_dnnmAA(self): self.check2dMatrixBlocks("dnnmAA")

    def test2dMatrixBlocks_dnnaLl(self): self.check2dMatrixBlocks("dnnaLl")
    def test2dMatrixBlocks_dnnaLL(self): self.check2dMatrixBlocks("dnnaLL")
    def test2dMatrixBlocks_dnnaLB(self): self.check2dMatrixBlocks("dnnaLB")
    def test2dMatrixBlocks_dnnaLA(self): self.check2dMatrixBlocks("dnnaLA")

    def test2dMatrixBlocks_dnnaAl(self): self.check2dMatrixBlocks("dnnaAl")
    def test2dMatrixBlocks_dnnaAL(self): self.check2dMatrixBlocks("dnnaAL")
    def test2dMatrixBlocks_dnnaAB(self): self.check2dMatrixBlocks("dnnaAB")
    def test2dMatrixBlocks_dnnaAA(self): self.check2dMatrixBlocks("dnnaAA")

    def test2dMatrixBlocks_dnnlLl(self): self.check2dMatrixBlocks("dnnlLl")
    def test2dMatrixBlocks_dnnlLL(self): self.check2dMatrixBlocks("dnnlLL")
    def test2dMatrixBlocks_dnnlLB(self): self.check2dMatrixBlocks("dnnlLB")
    def test2dMatrixBlocks_dnnlLA(self): self.check2dMatrixBlocks("dnnlLA")

    def test2dMatrixBlocks_dnnlAl(self): self.check2dMatrixBlocks("dnnlAl")
    def test2dMatrixBlocks_dnnlAL(self): self.check2dMatrixBlocks("dnnlAL")
    def test2dMatrixBlocks_dnnlAB(self): self.check2dMatrixBlocks("dnnlAB")
    def test2dMatrixBlocks_dnnlAA(self): self.check2dMatrixBlocks("dnnlAA")

    def test2dMatrixBlocks_dnnLLl(self): self.check2dMatrixBlocks("dnnLLl")
    def test2dMatrixBlocks_dnnLLL(self): self.check2dMatrixBlocks("dnnLLL")
    def test2dMatrixBlocks_dnnLLB(self): self.check2dMatrixBlocks("dnnLLB")
    def test2dMatrixBlocks_dnnLLA(self): self.check2dMatrixBlocks("dnnLLA")

    def test2dMatrixBlocks_dnnLAl(self): self.check2dMatrixBlocks("dnnLAl")
    def test2dMatrixBlocks_dnnLAL(self): self.check2dMatrixBlocks("dnnLAL")
    def test2dMatrixBlocks_dnnLAB(self): self.check2dMatrixBlocks("dnnLAB")
    def test2dMatrixBlocks_dnnLAA(self): self.check2dMatrixBlocks("dnnLAA")

    def test2dMatrixBlocks_dnnALl(self): self.check2dMatrixBlocks("dnnALl")
    def test2dMatrixBlocks_dnnALL(self): self.check2dMatrixBlocks("dnnALL")
    def test2dMatrixBlocks_dnnALB(self): self.check2dMatrixBlocks("dnnALB")
    def test2dMatrixBlocks_dnnALA(self): self.check2dMatrixBlocks("dnnALA")

    def test2dMatrixBlocks_dnnAAl(self): self.check2dMatrixBlocks("dnnAAl")
    def test2dMatrixBlocks_dnnAAL(self): self.check2dMatrixBlocks("dnnAAL")
    def test2dMatrixBlocks_dnnAAB(self): self.check2dMatrixBlocks("dnnAAB")
    def test2dMatrixBlocks_dnnAAA(self): self.check2dMatrixBlocks("dnnAAA")

    def test2dMatrixBlocks_dnNmLl(self): self.check2dMatrixBlocks("dnNmLl")
    def test2dMatrixBlocks_dnNmLL(self): self.check2dMatrixBlocks("dnNmLL")
    def test2dMatrixBlocks_dnNmLB(self): self.check2dMatrixBlocks("dnNmLB")
    def test2dMatrixBlocks_dnNmLA(self): self.check2dMatrixBlocks("dnNmLA")

    def test2dMatrixBlocks_dnNmAl(self): self.check2dMatrixBlocks("dnNmAl")
    def test2dMatrixBlocks_dnNmAL(self): self.check2dMatrixBlocks("dnNmAL")
    def test2dMatrixBlocks_dnNmAB(self): self.check2dMatrixBlocks("dnNmAB")
    def test2dMatrixBlocks_dnNmAA(self): self.check2dMatrixBlocks("dnNmAA")

    def test2dMatrixBlocks_dnNaLl(self): self.check2dMatrixBlocks("dnNaLl")
    def test2dMatrixBlocks_dnNaLL(self): self.check2dMatrixBlocks("dnNaLL")
    def test2dMatrixBlocks_dnNaLB(self): self.check2dMatrixBlocks("dnNaLB")
    def test2dMatrixBlocks_dnNaLA(self): self.check2dMatrixBlocks("dnNaLA")

    def test2dMatrixBlocks_dnNaAl(self): self.check2dMatrixBlocks("dnNaAl")
    def test2dMatrixBlocks_dnNaAL(self): self.check2dMatrixBlocks("dnNaAL")
    def test2dMatrixBlocks_dnNaAB(self): self.check2dMatrixBlocks("dnNaAB")
    def test2dMatrixBlocks_dnNaAA(self): self.check2dMatrixBlocks("dnNaAA")

    def test2dMatrixBlocks_dnNlLl(self): self.check2dMatrixBlocks("dnNlLl")
    def test2dMatrixBlocks_dnNlLL(self): self.check2dMatrixBlocks("dnNlLL")
    def test2dMatrixBlocks_dnNlLB(self): self.check2dMatrixBlocks("dnNlLB")
    def test2dMatrixBlocks_dnNlLA(self): self.check2dMatrixBlocks("dnNlLA")

    def test2dMatrixBlocks_dnNlAl(self): self.check2dMatrixBlocks("dnNlAl")
    def test2dMatrixBlocks_dnNlAL(self): self.check2dMatrixBlocks("dnNlAL")
    def test2dMatrixBlocks_dnNlAB(self): self.check2dMatrixBlocks("dnNlAB")
    def test2dMatrixBlocks_dnNlAA(self): self.check2dMatrixBlocks("dnNlAA")

    def test2dMatrixBlocks_dnNLLl(self): self.check2dMatrixBlocks("dnNLLl")
    def test2dMatrixBlocks_dnNLLL(self): self.check2dMatrixBlocks("dnNLLL")
    def test2dMatrixBlocks_dnNLLB(self): self.check2dMatrixBlocks("dnNLLB")
    def test2dMatrixBlocks_dnNLLA(self): self.check2dMatrixBlocks("dnNLLA")

    def test2dMatrixBlocks_dnNLAl(self): self.check2dMatrixBlocks("dnNLAl")
    def test2dMatrixBlocks_dnNLAL(self): self.check2dMatrixBlocks("dnNLAL")
    def test2dMatrixBlocks_dnNLAB(self): self.check2dMatrixBlocks("dnNLAB")
    def test2dMatrixBlocks_dnNLAA(self): self.check2dMatrixBlocks("dnNLAA")

    def test2dMatrixBlocks_dnNALl(self): self.check2dMatrixBlocks("dnNALl")
    def test2dMatrixBlocks_dnNALL(self): self.check2dMatrixBlocks("dnNALL")
    def test2dMatrixBlocks_dnNALB(self): self.check2dMatrixBlocks("dnNALB")
    def test2dMatrixBlocks_dnNALA(self): self.check2dMatrixBlocks("dnNALA")

    def test2dMatrixBlocks_dnNAAl(self): self.check2dMatrixBlocks("dnNAAl")
    def test2dMatrixBlocks_dnNAAL(self): self.check2dMatrixBlocks("dnNAAL")
    def test2dMatrixBlocks_dnNAAB(self): self.check2dMatrixBlocks("dnNAAB")
    def test2dMatrixBlocks_dnNAAA(self): self.check2dMatrixBlocks("dnNAAA")

    def test2dMatrixBlocks_dNtsLl(self): self.check2dMatrixBlocks("dNtsLl")
    def test2dMatrixBlocks_dNtsLL(self): self.check2dMatrixBlocks("dNtsLL")
    def test2dMatrixBlocks_dNtsLB(self): self.check2dMatrixBlocks("dNtsLB")
    def test2dMatrixBlocks_dNtsLA(self): self.check2dMatrixBlocks("dNtsLA")

    def test2dMatrixBlocks_dNtsAl(self): self.check2dMatrixBlocks("dNtsAl")
    def test2dMatrixBlocks_dNtsAL(self): self.check2dMatrixBlocks("dNtsAL")
    def test2dMatrixBlocks_dNtsAB(self): self.check2dMatrixBlocks("dNtsAB")
    def test2dMatrixBlocks_dNtsAA(self): self.check2dMatrixBlocks("dNtsAA")

    def test2dMatrixBlocks_dNtmLl(self): self.check2dMatrixBlocks("dNtmLl")
    def test2dMatrixBlocks_dNtmLL(self): self.check2dMatrixBlocks("dNtmLL")
    def test2dMatrixBlocks_dNtmLB(self): self.check2dMatrixBlocks("dNtmLB")
    def test2dMatrixBlocks_dNtmLA(self): self.check2dMatrixBlocks("dNtmLA")

    def test2dMatrixBlocks_dNtmAl(self): self.check2dMatrixBlocks("dNtmAl")
    def test2dMatrixBlocks_dNtmAL(self): self.check2dMatrixBlocks("dNtmAL")
    def test2dMatrixBlocks_dNtmAB(self): self.check2dMatrixBlocks("dNtmAB")
    def test2dMatrixBlocks_dNtmAA(self): self.check2dMatrixBlocks("dNtmAA")

    def test2dMatrixBlocks_dNtaLl(self): self.check2dMatrixBlocks("dNtaLl")
    def test2dMatrixBlocks_dNtaLL(self): self.check2dMatrixBlocks("dNtaLL")
    def test2dMatrixBlocks_dNtaLB(self): self.check2dMatrixBlocks("dNtaLB")
    def test2dMatrixBlocks_dNtaLA(self): self.check2dMatrixBlocks("dNtaLA")

    def test2dMatrixBlocks_dNtaAl(self): self.check2dMatrixBlocks("dNtaAl")
    def test2dMatrixBlocks_dNtaAL(self): self.check2dMatrixBlocks("dNtaAL")
    def test2dMatrixBlocks_dNtaAB(self): self.check2dMatrixBlocks("dNtaAB")
    def test2dMatrixBlocks_dNtaAA(self): self.check2dMatrixBlocks("dNtaAA")

    def test2dMatrixBlocks_dNtlLl(self): self.check2dMatrixBlocks("dNtlLl")
    def test2dMatrixBlocks_dNtlLL(self): self.check2dMatrixBlocks("dNtlLL")
    def test2dMatrixBlocks_dNtlLB(self): self.check2dMatrixBlocks("dNtlLB")
    def test2dMatrixBlocks_dNtlLA(self): self.check2dMatrixBlocks("dNtlLA")

    def test2dMatrixBlocks_dNtlAl(self): self.check2dMatrixBlocks("dNtlAl")
    def test2dMatrixBlocks_dNtlAL(self): self.check2dMatrixBlocks("dNtlAL")
    def test2dMatrixBlocks_dNtlAB(self): self.check2dMatrixBlocks("dNtlAB")
    def test2dMatrixBlocks_dNtlAA(self): self.check2dMatrixBlocks("dNtlAA")

    def test2dMatrixBlocks_dNtLLl(self): self.check2dMatrixBlocks("dNtLLl")
    def test2dMatrixBlocks_dNtLLL(self): self.check2dMatrixBlocks("dNtLLL")
    def test2dMatrixBlocks_dNtLLB(self): self.check2dMatrixBlocks("dNtLLB")
    def test2dMatrixBlocks_dNtLLA(self): self.check2dMatrixBlocks("dNtLLA")

    def test2dMatrixBlocks_dNtLAl(self): self.check2dMatrixBlocks("dNtLAl")
    def test2dMatrixBlocks_dNtLAL(self): self.check2dMatrixBlocks("dNtLAL")
    def test2dMatrixBlocks_dNtLAB(self): self.check2dMatrixBlocks("dNtLAB")
    def test2dMatrixBlocks_dNtLAA(self): self.check2dMatrixBlocks("dNtLAA")

    def test2dMatrixBlocks_dNtALl(self): self.check2dMatrixBlocks("dNtALl")
    def test2dMatrixBlocks_dNtALL(self): self.check2dMatrixBlocks("dNtALL")
    def test2dMatrixBlocks_dNtALB(self): self.check2dMatrixBlocks("dNtALB")
    def test2dMatrixBlocks_dNtALA(self): self.check2dMatrixBlocks("dNtALA")

    def test2dMatrixBlocks_dNtAAl(self): self.check2dMatrixBlocks("dNtAAl")
    def test2dMatrixBlocks_dNtAAL(self): self.check2dMatrixBlocks("dNtAAL")
    def test2dMatrixBlocks_dNtAAB(self): self.check2dMatrixBlocks("dNtAAB")
    def test2dMatrixBlocks_dNtAAA(self): self.check2dMatrixBlocks("dNtAAA")

    def test2dMatrixBlocks_dNnsLl(self): self.check2dMatrixBlocks("dNnsLl")
    def test2dMatrixBlocks_dNnsLL(self): self.check2dMatrixBlocks("dNnsLL")
    def test2dMatrixBlocks_dNnsLB(self): self.check2dMatrixBlocks("dNnsLB")
    def test2dMatrixBlocks_dNnsLA(self): self.check2dMatrixBlocks("dNnsLA")

    def test2dMatrixBlocks_dNnsAl(self): self.check2dMatrixBlocks("dNnsAl")
    def test2dMatrixBlocks_dNnsAL(self): self.check2dMatrixBlocks("dNnsAL")
    def test2dMatrixBlocks_dNnsAB(self): self.check2dMatrixBlocks("dNnsAB")
    def test2dMatrixBlocks_dNnsAA(self): self.check2dMatrixBlocks("dNnsAA")

    def test2dMatrixBlocks_dNnmLl(self): self.check2dMatrixBlocks("dNnmLl")
    def test2dMatrixBlocks_dNnmLL(self): self.check2dMatrixBlocks("dNnmLL")
    def test2dMatrixBlocks_dNnmLB(self): self.check2dMatrixBlocks("dNnmLB")
    def test2dMatrixBlocks_dNnmLA(self): self.check2dMatrixBlocks("dNnmLA")

    def test2dMatrixBlocks_dNnmAl(self): self.check2dMatrixBlocks("dNnmAl")
    def test2dMatrixBlocks_dNnmAL(self): self.check2dMatrixBlocks("dNnmAL")
    def test2dMatrixBlocks_dNnmAB(self): self.check2dMatrixBlocks("dNnmAB")
    def test2dMatrixBlocks_dNnmAA(self): self.check2dMatrixBlocks("dNnmAA")

    def test2dMatrixBlocks_dNnaLl(self): self.check2dMatrixBlocks("dNnaLl")
    def test2dMatrixBlocks_dNnaLL(self): self.check2dMatrixBlocks("dNnaLL")
    def test2dMatrixBlocks_dNnaLB(self): self.check2dMatrixBlocks("dNnaLB")
    def test2dMatrixBlocks_dNnaLA(self): self.check2dMatrixBlocks("dNnaLA")

    def test2dMatrixBlocks_dNnaAl(self): self.check2dMatrixBlocks("dNnaAl")
    def test2dMatrixBlocks_dNnaAL(self): self.check2dMatrixBlocks("dNnaAL")
    def test2dMatrixBlocks_dNnaAB(self): self.check2dMatrixBlocks("dNnaAB")
    def test2dMatrixBlocks_dNnaAA(self): self.check2dMatrixBlocks("dNnaAA")

    def test2dMatrixBlocks_dNnlLl(self): self.check2dMatrixBlocks("dNnlLl")
    def test2dMatrixBlocks_dNnlLL(self): self.check2dMatrixBlocks("dNnlLL")
    def test2dMatrixBlocks_dNnlLB(self): self.check2dMatrixBlocks("dNnlLB")
    def test2dMatrixBlocks_dNnlLA(self): self.check2dMatrixBlocks("dNnlLA")

    def test2dMatrixBlocks_dNnlAl(self): self.check2dMatrixBlocks("dNnlAl")
    def test2dMatrixBlocks_dNnlAL(self): self.check2dMatrixBlocks("dNnlAL")
    def test2dMatrixBlocks_dNnlAB(self): self.check2dMatrixBlocks("dNnlAB")
    def test2dMatrixBlocks_dNnlAA(self): self.check2dMatrixBlocks("dNnlAA")

    def test2dMatrixBlocks_dNnLLl(self): self.check2dMatrixBlocks("dNnLLl")
    def test2dMatrixBlocks_dNnLLL(self): self.check2dMatrixBlocks("dNnLLL")
    def test2dMatrixBlocks_dNnLLB(self): self.check2dMatrixBlocks("dNnLLB")
    def test2dMatrixBlocks_dNnLLA(self): self.check2dMatrixBlocks("dNnLLA")

    def test2dMatrixBlocks_dNnLAl(self): self.check2dMatrixBlocks("dNnLAl")
    def test2dMatrixBlocks_dNnLAL(self): self.check2dMatrixBlocks("dNnLAL")
    def test2dMatrixBlocks_dNnLAB(self): self.check2dMatrixBlocks("dNnLAB")
    def test2dMatrixBlocks_dNnLAA(self): self.check2dMatrixBlocks("dNnLAA")

    def test2dMatrixBlocks_dNnALl(self): self.check2dMatrixBlocks("dNnALl")
    def test2dMatrixBlocks_dNnALL(self): self.check2dMatrixBlocks("dNnALL")
    def test2dMatrixBlocks_dNnALB(self): self.check2dMatrixBlocks("dNnALB")
    def test2dMatrixBlocks_dNnALA(self): self.check2dMatrixBlocks("dNnALA")

    def test2dMatrixBlocks_dNnAAl(self): self.check2dMatrixBlocks("dNnAAl")
    def test2dMatrixBlocks_dNnAAL(self): self.check2dMatrixBlocks("dNnAAL")
    def test2dMatrixBlocks_dNnAAB(self): self.check2dMatrixBlocks("dNnAAB")
    def test2dMatrixBlocks_dNnAAA(self): self.check2dMatrixBlocks("dNnAAA")

    def test2dMatrixBlocks_dNNsLl(self): self.check2dMatrixBlocks("dNNsLl")
    def test2dMatrixBlocks_dNNsLL(self): self.check2dMatrixBlocks("dNNsLL")
    def test2dMatrixBlocks_dNNsLB(self): self.check2dMatrixBlocks("dNNsLB")
    def test2dMatrixBlocks_dNNsLA(self): self.check2dMatrixBlocks("dNNsLA")

    def test2dMatrixBlocks_dNNsAl(self): self.check2dMatrixBlocks("dNNsAl")
    def test2dMatrixBlocks_dNNsAL(self): self.check2dMatrixBlocks("dNNsAL")
    def test2dMatrixBlocks_dNNsAB(self): self.check2dMatrixBlocks("dNNsAB")
    def test2dMatrixBlocks_dNNsAA(self): self.check2dMatrixBlocks("dNNsAA")

    def test2dMatrixBlocks_dNNmLl(self): self.check2dMatrixBlocks("dNNmLl")
    def test2dMatrixBlocks_dNNmLL(self): self.check2dMatrixBlocks("dNNmLL")
    def test2dMatrixBlocks_dNNmLB(self): self.check2dMatrixBlocks("dNNmLB")
    def test2dMatrixBlocks_dNNmLA(self): self.check2dMatrixBlocks("dNNmLA")

    def test2dMatrixBlocks_dNNmAl(self): self.check2dMatrixBlocks("dNNmAl")
    def test2dMatrixBlocks_dNNmAL(self): self.check2dMatrixBlocks("dNNmAL")
    def test2dMatrixBlocks_dNNmAB(self): self.check2dMatrixBlocks("dNNmAB")
    def test2dMatrixBlocks_dNNmAA(self): self.check2dMatrixBlocks("dNNmAA")

    def test2dMatrixBlocks_dNNaLl(self): self.check2dMatrixBlocks("dNNaLl")
    def test2dMatrixBlocks_dNNaLL(self): self.check2dMatrixBlocks("dNNaLL")
    def test2dMatrixBlocks_dNNaLB(self): self.check2dMatrixBlocks("dNNaLB")
    def test2dMatrixBlocks_dNNaLA(self): self.check2dMatrixBlocks("dNNaLA")

    def test2dMatrixBlocks_dNNaAl(self): self.check2dMatrixBlocks("dNNaAl")
    def test2dMatrixBlocks_dNNaAL(self): self.check2dMatrixBlocks("dNNaAL")
    def test2dMatrixBlocks_dNNaAB(self): self.check2dMatrixBlocks("dNNaAB")
    def test2dMatrixBlocks_dNNaAA(self): self.check2dMatrixBlocks("dNNaAA")

    def test2dMatrixBlocks_dNNlLl(self): self.check2dMatrixBlocks("dNNlLl")
    def test2dMatrixBlocks_dNNlLL(self): self.check2dMatrixBlocks("dNNlLL")
    def test2dMatrixBlocks_dNNlLB(self): self.check2dMatrixBlocks("dNNlLB")
    def test2dMatrixBlocks_dNNlLA(self): self.check2dMatrixBlocks("dNNlLA")

    def test2dMatrixBlocks_dNNlAl(self): self.check2dMatrixBlocks("dNNlAl")
    def test2dMatrixBlocks_dNNlAL(self): self.check2dMatrixBlocks("dNNlAL")
    def test2dMatrixBlocks_dNNlAB(self): self.check2dMatrixBlocks("dNNlAB")
    def test2dMatrixBlocks_dNNlAA(self): self.check2dMatrixBlocks("dNNlAA")

    def test2dMatrixBlocks_dNNLLl(self): self.check2dMatrixBlocks("dNNLLl")
    def test2dMatrixBlocks_dNNLLL(self): self.check2dMatrixBlocks("dNNLLL")
    def test2dMatrixBlocks_dNNLLB(self): self.check2dMatrixBlocks("dNNLLB")
    def test2dMatrixBlocks_dNNLLA(self): self.check2dMatrixBlocks("dNNLLA")

    def test2dMatrixBlocks_dNNLAl(self): self.check2dMatrixBlocks("dNNLAl")
    def test2dMatrixBlocks_dNNLAL(self): self.check2dMatrixBlocks("dNNLAL")
    def test2dMatrixBlocks_dNNLAB(self): self.check2dMatrixBlocks("dNNLAB")
    def test2dMatrixBlocks_dNNLAA(self): self.check2dMatrixBlocks("dNNLAA")

    def test2dMatrixBlocks_dNNALl(self): self.check2dMatrixBlocks("dNNALl")
    def test2dMatrixBlocks_dNNALL(self): self.check2dMatrixBlocks("dNNALL")
    def test2dMatrixBlocks_dNNALB(self): self.check2dMatrixBlocks("dNNALB")
    def test2dMatrixBlocks_dNNALA(self): self.check2dMatrixBlocks("dNNALA")

    def test2dMatrixBlocks_dNNAAl(self): self.check2dMatrixBlocks("dNNAAl")
    def test2dMatrixBlocks_dNNAAL(self): self.check2dMatrixBlocks("dNNAAL")
    def test2dMatrixBlocks_dNNAAB(self): self.check2dMatrixBlocks("dNNAAB")
    def test2dMatrixBlocks_dNNAAA(self): self.check2dMatrixBlocks("dNNAAA")

    def test2dMatrixBlocks_lttsLl(self): self.check2dMatrixBlocks("lttsLl")
    def test2dMatrixBlocks_lttsLL(self): self.check2dMatrixBlocks("lttsLL")
    def test2dMatrixBlocks_lttsLB(self): self.check2dMatrixBlocks("lttsLB")
    def test2dMatrixBlocks_lttsLA(self): self.check2dMatrixBlocks("lttsLA")

    def test2dMatrixBlocks_lttsAl(self): self.check2dMatrixBlocks("lttsAl")
    def test2dMatrixBlocks_lttsAL(self): self.check2dMatrixBlocks("lttsAL")
    def test2dMatrixBlocks_lttsAB(self): self.check2dMatrixBlocks("lttsAB")
    def test2dMatrixBlocks_lttsAA(self): self.check2dMatrixBlocks("lttsAA")

    def test2dMatrixBlocks_lttmLl(self): self.check2dMatrixBlocks("lttmLl")
    def test2dMatrixBlocks_lttmLL(self): self.check2dMatrixBlocks("lttmLL")
    def test2dMatrixBlocks_lttmLB(self): self.check2dMatrixBlocks("lttmLB")
    def test2dMatrixBlocks_lttmLA(self): self.check2dMatrixBlocks("lttmLA")

    def test2dMatrixBlocks_lttmAl(self): self.check2dMatrixBlocks("lttmAl")
    def test2dMatrixBlocks_lttmAL(self): self.check2dMatrixBlocks("lttmAL")
    def test2dMatrixBlocks_lttmAB(self): self.check2dMatrixBlocks("lttmAB")
    def test2dMatrixBlocks_lttmAA(self): self.check2dMatrixBlocks("lttmAA")

    def test2dMatrixBlocks_lttaLl(self): self.check2dMatrixBlocks("lttaLl")
    def test2dMatrixBlocks_lttaLL(self): self.check2dMatrixBlocks("lttaLL")
    def test2dMatrixBlocks_lttaLB(self): self.check2dMatrixBlocks("lttaLB")
    def test2dMatrixBlocks_lttaLA(self): self.check2dMatrixBlocks("lttaLA")

    def test2dMatrixBlocks_lttaAl(self): self.check2dMatrixBlocks("lttaAl")
    def test2dMatrixBlocks_lttaAL(self): self.check2dMatrixBlocks("lttaAL")
    def test2dMatrixBlocks_lttaAB(self): self.check2dMatrixBlocks("lttaAB")
    def test2dMatrixBlocks_lttaAA(self): self.check2dMatrixBlocks("lttaAA")

    def test2dMatrixBlocks_lttlLl(self): self.check2dMatrixBlocks("lttlLl")
    def test2dMatrixBlocks_lttlLL(self): self.check2dMatrixBlocks("lttlLL")
    def test2dMatrixBlocks_lttlLB(self): self.check2dMatrixBlocks("lttlLB")
    def test2dMatrixBlocks_lttlLA(self): self.check2dMatrixBlocks("lttlLA")

    def test2dMatrixBlocks_lttlAl(self): self.check2dMatrixBlocks("lttlAl")
    def test2dMatrixBlocks_lttlAL(self): self.check2dMatrixBlocks("lttlAL")
    def test2dMatrixBlocks_lttlAB(self): self.check2dMatrixBlocks("lttlAB")
    def test2dMatrixBlocks_lttlAA(self): self.check2dMatrixBlocks("lttlAA")

    def test2dMatrixBlocks_lttLLl(self): self.check2dMatrixBlocks("lttLLl")
    def test2dMatrixBlocks_lttLLL(self): self.check2dMatrixBlocks("lttLLL")
    def test2dMatrixBlocks_lttLLB(self): self.check2dMatrixBlocks("lttLLB")
    def test2dMatrixBlocks_lttLLA(self): self.check2dMatrixBlocks("lttLLA")

    def test2dMatrixBlocks_lttLAl(self): self.check2dMatrixBlocks("lttLAl")
    def test2dMatrixBlocks_lttLAL(self): self.check2dMatrixBlocks("lttLAL")
    def test2dMatrixBlocks_lttLAB(self): self.check2dMatrixBlocks("lttLAB")
    def test2dMatrixBlocks_lttLAA(self): self.check2dMatrixBlocks("lttLAA")

    def test2dMatrixBlocks_lttALl(self): self.check2dMatrixBlocks("lttALl")
    def test2dMatrixBlocks_lttALL(self): self.check2dMatrixBlocks("lttALL")
    def test2dMatrixBlocks_lttALB(self): self.check2dMatrixBlocks("lttALB")
    def test2dMatrixBlocks_lttALA(self): self.check2dMatrixBlocks("lttALA")

    def test2dMatrixBlocks_lttAAl(self): self.check2dMatrixBlocks("lttAAl")
    def test2dMatrixBlocks_lttAAL(self): self.check2dMatrixBlocks("lttAAL")
    def test2dMatrixBlocks_lttAAB(self): self.check2dMatrixBlocks("lttAAB")
    def test2dMatrixBlocks_lttAAA(self): self.check2dMatrixBlocks("lttAAA")

    def test2dMatrixBlocks_ltnsLl(self): self.check2dMatrixBlocks("ltnsLl")
    def test2dMatrixBlocks_ltnsLL(self): self.check2dMatrixBlocks("ltnsLL")
    def test2dMatrixBlocks_ltnsLB(self): self.check2dMatrixBlocks("ltnsLB")
    def test2dMatrixBlocks_ltnsLA(self): self.check2dMatrixBlocks("ltnsLA")

    def test2dMatrixBlocks_ltnsAl(self): self.check2dMatrixBlocks("ltnsAl")
    def test2dMatrixBlocks_ltnsAL(self): self.check2dMatrixBlocks("ltnsAL")
    def test2dMatrixBlocks_ltnsAB(self): self.check2dMatrixBlocks("ltnsAB")
    def test2dMatrixBlocks_ltnsAA(self): self.check2dMatrixBlocks("ltnsAA")

    def test2dMatrixBlocks_ltnmLl(self): self.check2dMatrixBlocks("ltnmLl")
    def test2dMatrixBlocks_ltnmLL(self): self.check2dMatrixBlocks("ltnmLL")
    def test2dMatrixBlocks_ltnmLB(self): self.check2dMatrixBlocks("ltnmLB")
    def test2dMatrixBlocks_ltnmLA(self): self.check2dMatrixBlocks("ltnmLA")

    def test2dMatrixBlocks_ltnmAl(self): self.check2dMatrixBlocks("ltnmAl")
    def test2dMatrixBlocks_ltnmAL(self): self.check2dMatrixBlocks("ltnmAL")
    def test2dMatrixBlocks_ltnmAB(self): self.check2dMatrixBlocks("ltnmAB")
    def test2dMatrixBlocks_ltnmAA(self): self.check2dMatrixBlocks("ltnmAA")

    def test2dMatrixBlocks_ltnaLl(self): self.check2dMatrixBlocks("ltnaLl")
    def test2dMatrixBlocks_ltnaLL(self): self.check2dMatrixBlocks("ltnaLL")
    def test2dMatrixBlocks_ltnaLB(self): self.check2dMatrixBlocks("ltnaLB")
    def test2dMatrixBlocks_ltnaLA(self): self.check2dMatrixBlocks("ltnaLA")

    def test2dMatrixBlocks_ltnaAl(self): self.check2dMatrixBlocks("ltnaAl")
    def test2dMatrixBlocks_ltnaAL(self): self.check2dMatrixBlocks("ltnaAL")
    def test2dMatrixBlocks_ltnaAB(self): self.check2dMatrixBlocks("ltnaAB")
    def test2dMatrixBlocks_ltnaAA(self): self.check2dMatrixBlocks("ltnaAA")

    def test2dMatrixBlocks_ltnlLl(self): self.check2dMatrixBlocks("ltnlLl")
    def test2dMatrixBlocks_ltnlLL(self): self.check2dMatrixBlocks("ltnlLL")
    def test2dMatrixBlocks_ltnlLB(self): self.check2dMatrixBlocks("ltnlLB")
    def test2dMatrixBlocks_ltnlLA(self): self.check2dMatrixBlocks("ltnlLA")

    def test2dMatrixBlocks_ltnlAl(self): self.check2dMatrixBlocks("ltnlAl")
    def test2dMatrixBlocks_ltnlAL(self): self.check2dMatrixBlocks("ltnlAL")
    def test2dMatrixBlocks_ltnlAB(self): self.check2dMatrixBlocks("ltnlAB")
    def test2dMatrixBlocks_ltnlAA(self): self.check2dMatrixBlocks("ltnlAA")

    def test2dMatrixBlocks_ltnLLl(self): self.check2dMatrixBlocks("ltnLLl")
    def test2dMatrixBlocks_ltnLLL(self): self.check2dMatrixBlocks("ltnLLL")
    def test2dMatrixBlocks_ltnLLB(self): self.check2dMatrixBlocks("ltnLLB")
    def test2dMatrixBlocks_ltnLLA(self): self.check2dMatrixBlocks("ltnLLA")

    def test2dMatrixBlocks_ltnLAl(self): self.check2dMatrixBlocks("ltnLAl")
    def test2dMatrixBlocks_ltnLAL(self): self.check2dMatrixBlocks("ltnLAL")
    def test2dMatrixBlocks_ltnLAB(self): self.check2dMatrixBlocks("ltnLAB")
    def test2dMatrixBlocks_ltnLAA(self): self.check2dMatrixBlocks("ltnLAA")

    def test2dMatrixBlocks_ltnALl(self): self.check2dMatrixBlocks("ltnALl")
    def test2dMatrixBlocks_ltnALL(self): self.check2dMatrixBlocks("ltnALL")
    def test2dMatrixBlocks_ltnALB(self): self.check2dMatrixBlocks("ltnALB")
    def test2dMatrixBlocks_ltnALA(self): self.check2dMatrixBlocks("ltnALA")

    def test2dMatrixBlocks_ltnAAl(self): self.check2dMatrixBlocks("ltnAAl")
    def test2dMatrixBlocks_ltnAAL(self): self.check2dMatrixBlocks("ltnAAL")
    def test2dMatrixBlocks_ltnAAB(self): self.check2dMatrixBlocks("ltnAAB")
    def test2dMatrixBlocks_ltnAAA(self): self.check2dMatrixBlocks("ltnAAA")

    def test2dMatrixBlocks_ltNsLl(self): self.check2dMatrixBlocks("ltNsLl")
    def test2dMatrixBlocks_ltNsLL(self): self.check2dMatrixBlocks("ltNsLL")
    def test2dMatrixBlocks_ltNsLB(self): self.check2dMatrixBlocks("ltNsLB")
    def test2dMatrixBlocks_ltNsLA(self): self.check2dMatrixBlocks("ltNsLA")

    def test2dMatrixBlocks_ltNsAl(self): self.check2dMatrixBlocks("ltNsAl")
    def test2dMatrixBlocks_ltNsAL(self): self.check2dMatrixBlocks("ltNsAL")
    def test2dMatrixBlocks_ltNsAB(self): self.check2dMatrixBlocks("ltNsAB")
    def test2dMatrixBlocks_ltNsAA(self): self.check2dMatrixBlocks("ltNsAA")

    def test2dMatrixBlocks_ltNmLl(self): self.check2dMatrixBlocks("ltNmLl")
    def test2dMatrixBlocks_ltNmLL(self): self.check2dMatrixBlocks("ltNmLL")
    def test2dMatrixBlocks_ltNmLB(self): self.check2dMatrixBlocks("ltNmLB")
    def test2dMatrixBlocks_ltNmLA(self): self.check2dMatrixBlocks("ltNmLA")

    def test2dMatrixBlocks_ltNmAl(self): self.check2dMatrixBlocks("ltNmAl")
    def test2dMatrixBlocks_ltNmAL(self): self.check2dMatrixBlocks("ltNmAL")
    def test2dMatrixBlocks_ltNmAB(self): self.check2dMatrixBlocks("ltNmAB")
    def test2dMatrixBlocks_ltNmAA(self): self.check2dMatrixBlocks("ltNmAA")

    def test2dMatrixBlocks_ltNaLl(self): self.check2dMatrixBlocks("ltNaLl")
    def test2dMatrixBlocks_ltNaLL(self): self.check2dMatrixBlocks("ltNaLL")
    def test2dMatrixBlocks_ltNaLB(self): self.check2dMatrixBlocks("ltNaLB")
    def test2dMatrixBlocks_ltNaLA(self): self.check2dMatrixBlocks("ltNaLA")

    def test2dMatrixBlocks_ltNaAl(self): self.check2dMatrixBlocks("ltNaAl")
    def test2dMatrixBlocks_ltNaAL(self): self.check2dMatrixBlocks("ltNaAL")
    def test2dMatrixBlocks_ltNaAB(self): self.check2dMatrixBlocks("ltNaAB")
    def test2dMatrixBlocks_ltNaAA(self): self.check2dMatrixBlocks("ltNaAA")

    def test2dMatrixBlocks_ltNlLl(self): self.check2dMatrixBlocks("ltNlLl")
    def test2dMatrixBlocks_ltNlLL(self): self.check2dMatrixBlocks("ltNlLL")
    def test2dMatrixBlocks_ltNlLB(self): self.check2dMatrixBlocks("ltNlLB")
    def test2dMatrixBlocks_ltNlLA(self): self.check2dMatrixBlocks("ltNlLA")

    def test2dMatrixBlocks_ltNlAl(self): self.check2dMatrixBlocks("ltNlAl")
    def test2dMatrixBlocks_ltNlAL(self): self.check2dMatrixBlocks("ltNlAL")
    def test2dMatrixBlocks_ltNlAB(self): self.check2dMatrixBlocks("ltNlAB")
    def test2dMatrixBlocks_ltNlAA(self): self.check2dMatrixBlocks("ltNlAA")

    def test2dMatrixBlocks_ltNLLl(self): self.check2dMatrixBlocks("ltNLLl")
    def test2dMatrixBlocks_ltNLLL(self): self.check2dMatrixBlocks("ltNLLL")
    def test2dMatrixBlocks_ltNLLB(self): self.check2dMatrixBlocks("ltNLLB")
    def test2dMatrixBlocks_ltNLLA(self): self.check2dMatrixBlocks("ltNLLA")

    def test2dMatrixBlocks_ltNLAl(self): self.check2dMatrixBlocks("ltNLAl")
    def test2dMatrixBlocks_ltNLAL(self): self.check2dMatrixBlocks("ltNLAL")
    def test2dMatrixBlocks_ltNLAB(self): self.check2dMatrixBlocks("ltNLAB")
    def test2dMatrixBlocks_ltNLAA(self): self.check2dMatrixBlocks("ltNLAA")

    def test2dMatrixBlocks_ltNALl(self): self.check2dMatrixBlocks("ltNALl")
    def test2dMatrixBlocks_ltNALL(self): self.check2dMatrixBlocks("ltNALL")
    def test2dMatrixBlocks_ltNALB(self): self.check2dMatrixBlocks("ltNALB")
    def test2dMatrixBlocks_ltNALA(self): self.check2dMatrixBlocks("ltNALA")

    def test2dMatrixBlocks_ltNAAl(self): self.check2dMatrixBlocks("ltNAAl")
    def test2dMatrixBlocks_ltNAAL(self): self.check2dMatrixBlocks("ltNAAL")
    def test2dMatrixBlocks_ltNAAB(self): self.check2dMatrixBlocks("ltNAAB")
    def test2dMatrixBlocks_ltNAAA(self): self.check2dMatrixBlocks("ltNAAA")

    def test2dMatrixBlocks_ltlsLl(self): self.check2dMatrixBlocks("ltlsLl")
    def test2dMatrixBlocks_ltlsLL(self): self.check2dMatrixBlocks("ltlsLL")
    def test2dMatrixBlocks_ltlsLB(self): self.check2dMatrixBlocks("ltlsLB")
    def test2dMatrixBlocks_ltlsLA(self): self.check2dMatrixBlocks("ltlsLA")

    def test2dMatrixBlocks_ltlsAl(self): self.check2dMatrixBlocks("ltlsAl")
    def test2dMatrixBlocks_ltlsAL(self): self.check2dMatrixBlocks("ltlsAL")
    def test2dMatrixBlocks_ltlsAB(self): self.check2dMatrixBlocks("ltlsAB")
    def test2dMatrixBlocks_ltlsAA(self): self.check2dMatrixBlocks("ltlsAA")

    def test2dMatrixBlocks_ltlmLl(self): self.check2dMatrixBlocks("ltlmLl")
    def test2dMatrixBlocks_ltlmLL(self): self.check2dMatrixBlocks("ltlmLL")
    def test2dMatrixBlocks_ltlmLB(self): self.check2dMatrixBlocks("ltlmLB")
    def test2dMatrixBlocks_ltlmLA(self): self.check2dMatrixBlocks("ltlmLA")

    def test2dMatrixBlocks_ltlmAl(self): self.check2dMatrixBlocks("ltlmAl")
    def test2dMatrixBlocks_ltlmAL(self): self.check2dMatrixBlocks("ltlmAL")
    def test2dMatrixBlocks_ltlmAB(self): self.check2dMatrixBlocks("ltlmAB")
    def test2dMatrixBlocks_ltlmAA(self): self.check2dMatrixBlocks("ltlmAA")

    def test2dMatrixBlocks_ltlaLl(self): self.check2dMatrixBlocks("ltlaLl")
    def test2dMatrixBlocks_ltlaLL(self): self.check2dMatrixBlocks("ltlaLL")
    def test2dMatrixBlocks_ltlaLB(self): self.check2dMatrixBlocks("ltlaLB")
    def test2dMatrixBlocks_ltlaLA(self): self.check2dMatrixBlocks("ltlaLA")

    def test2dMatrixBlocks_ltlaAl(self): self.check2dMatrixBlocks("ltlaAl")
    def test2dMatrixBlocks_ltlaAL(self): self.check2dMatrixBlocks("ltlaAL")
    def test2dMatrixBlocks_ltlaAB(self): self.check2dMatrixBlocks("ltlaAB")
    def test2dMatrixBlocks_ltlaAA(self): self.check2dMatrixBlocks("ltlaAA")

    def test2dMatrixBlocks_ltllLl(self): self.check2dMatrixBlocks("ltllLl")
    def test2dMatrixBlocks_ltllLL(self): self.check2dMatrixBlocks("ltllLL")
    def test2dMatrixBlocks_ltllLB(self): self.check2dMatrixBlocks("ltllLB")
    def test2dMatrixBlocks_ltllLA(self): self.check2dMatrixBlocks("ltllLA")

    def test2dMatrixBlocks_ltllAl(self): self.check2dMatrixBlocks("ltllAl")
    def test2dMatrixBlocks_ltllAL(self): self.check2dMatrixBlocks("ltllAL")
    def test2dMatrixBlocks_ltllAB(self): self.check2dMatrixBlocks("ltllAB")
    def test2dMatrixBlocks_ltllAA(self): self.check2dMatrixBlocks("ltllAA")

    def test2dMatrixBlocks_ltlLLl(self): self.check2dMatrixBlocks("ltlLLl")
    def test2dMatrixBlocks_ltlLLL(self): self.check2dMatrixBlocks("ltlLLL")
    def test2dMatrixBlocks_ltlLLB(self): self.check2dMatrixBlocks("ltlLLB")
    def test2dMatrixBlocks_ltlLLA(self): self.check2dMatrixBlocks("ltlLLA")

    def test2dMatrixBlocks_ltlLAl(self): self.check2dMatrixBlocks("ltlLAl")
    def test2dMatrixBlocks_ltlLAL(self): self.check2dMatrixBlocks("ltlLAL")
    def test2dMatrixBlocks_ltlLAB(self): self.check2dMatrixBlocks("ltlLAB")
    def test2dMatrixBlocks_ltlLAA(self): self.check2dMatrixBlocks("ltlLAA")

    def test2dMatrixBlocks_ltlALl(self): self.check2dMatrixBlocks("ltlALl")
    def test2dMatrixBlocks_ltlALL(self): self.check2dMatrixBlocks("ltlALL")
    def test2dMatrixBlocks_ltlALB(self): self.check2dMatrixBlocks("ltlALB")
    def test2dMatrixBlocks_ltlALA(self): self.check2dMatrixBlocks("ltlALA")

    def test2dMatrixBlocks_ltlAAl(self): self.check2dMatrixBlocks("ltlAAl")
    def test2dMatrixBlocks_ltlAAL(self): self.check2dMatrixBlocks("ltlAAL")
    def test2dMatrixBlocks_ltlAAB(self): self.check2dMatrixBlocks("ltlAAB")
    def test2dMatrixBlocks_ltlAAA(self): self.check2dMatrixBlocks("ltlAAA")

    def test2dMatrixBlocks_lntmLl(self): self.check2dMatrixBlocks("lntmLl")
    def test2dMatrixBlocks_lntmLL(self): self.check2dMatrixBlocks("lntmLL")
    def test2dMatrixBlocks_lntmLB(self): self.check2dMatrixBlocks("lntmLB")
    def test2dMatrixBlocks_lntmLA(self): self.check2dMatrixBlocks("lntmLA")

    def test2dMatrixBlocks_lntmAl(self): self.check2dMatrixBlocks("lntmAl")
    def test2dMatrixBlocks_lntmAL(self): self.check2dMatrixBlocks("lntmAL")
    def test2dMatrixBlocks_lntmAB(self): self.check2dMatrixBlocks("lntmAB")
    def test2dMatrixBlocks_lntmAA(self): self.check2dMatrixBlocks("lntmAA")

    def test2dMatrixBlocks_lntaLl(self): self.check2dMatrixBlocks("lntaLl")
    def test2dMatrixBlocks_lntaLL(self): self.check2dMatrixBlocks("lntaLL")
    def test2dMatrixBlocks_lntaLB(self): self.check2dMatrixBlocks("lntaLB")
    def test2dMatrixBlocks_lntaLA(self): self.check2dMatrixBlocks("lntaLA")

    def test2dMatrixBlocks_lntaAl(self): self.check2dMatrixBlocks("lntaAl")
    def test2dMatrixBlocks_lntaAL(self): self.check2dMatrixBlocks("lntaAL")
    def test2dMatrixBlocks_lntaAB(self): self.check2dMatrixBlocks("lntaAB")
    def test2dMatrixBlocks_lntaAA(self): self.check2dMatrixBlocks("lntaAA")

    def test2dMatrixBlocks_lntlLl(self): self.check2dMatrixBlocks("lntlLl")
    def test2dMatrixBlocks_lntlLL(self): self.check2dMatrixBlocks("lntlLL")
    def test2dMatrixBlocks_lntlLB(self): self.check2dMatrixBlocks("lntlLB")
    def test2dMatrixBlocks_lntlLA(self): self.check2dMatrixBlocks("lntlLA")

    def test2dMatrixBlocks_lntlAl(self): self.check2dMatrixBlocks("lntlAl")
    def test2dMatrixBlocks_lntlAL(self): self.check2dMatrixBlocks("lntlAL")
    def test2dMatrixBlocks_lntlAB(self): self.check2dMatrixBlocks("lntlAB")
    def test2dMatrixBlocks_lntlAA(self): self.check2dMatrixBlocks("lntlAA")

    def test2dMatrixBlocks_lntLLl(self): self.check2dMatrixBlocks("lntLLl")
    def test2dMatrixBlocks_lntLLL(self): self.check2dMatrixBlocks("lntLLL")
    def test2dMatrixBlocks_lntLLB(self): self.check2dMatrixBlocks("lntLLB")
    def test2dMatrixBlocks_lntLLA(self): self.check2dMatrixBlocks("lntLLA")

    def test2dMatrixBlocks_lntLAl(self): self.check2dMatrixBlocks("lntLAl")
    def test2dMatrixBlocks_lntLAL(self): self.check2dMatrixBlocks("lntLAL")
    def test2dMatrixBlocks_lntLAB(self): self.check2dMatrixBlocks("lntLAB")
    def test2dMatrixBlocks_lntLAA(self): self.check2dMatrixBlocks("lntLAA")

    def test2dMatrixBlocks_lntALl(self): self.check2dMatrixBlocks("lntALl")
    def test2dMatrixBlocks_lntALL(self): self.check2dMatrixBlocks("lntALL")
    def test2dMatrixBlocks_lntALB(self): self.check2dMatrixBlocks("lntALB")
    def test2dMatrixBlocks_lntALA(self): self.check2dMatrixBlocks("lntALA")

    def test2dMatrixBlocks_lntAAl(self): self.check2dMatrixBlocks("lntAAl")
    def test2dMatrixBlocks_lntAAL(self): self.check2dMatrixBlocks("lntAAL")
    def test2dMatrixBlocks_lntAAB(self): self.check2dMatrixBlocks("lntAAB")
    def test2dMatrixBlocks_lntAAA(self): self.check2dMatrixBlocks("lntAAA")

    def test2dMatrixBlocks_lnnmLl(self): self.check2dMatrixBlocks("lnnmLl")
    def test2dMatrixBlocks_lnnmLL(self): self.check2dMatrixBlocks("lnnmLL")
    def test2dMatrixBlocks_lnnmLB(self): self.check2dMatrixBlocks("lnnmLB")
    def test2dMatrixBlocks_lnnmLA(self): self.check2dMatrixBlocks("lnnmLA")

    def test2dMatrixBlocks_lnnmAl(self): self.check2dMatrixBlocks("lnnmAl")
    def test2dMatrixBlocks_lnnmAL(self): self.check2dMatrixBlocks("lnnmAL")
    def test2dMatrixBlocks_lnnmAB(self): self.check2dMatrixBlocks("lnnmAB")
    def test2dMatrixBlocks_lnnmAA(self): self.check2dMatrixBlocks("lnnmAA")

    def test2dMatrixBlocks_lnnaLl(self): self.check2dMatrixBlocks("lnnaLl")
    def test2dMatrixBlocks_lnnaLL(self): self.check2dMatrixBlocks("lnnaLL")
    def test2dMatrixBlocks_lnnaLB(self): self.check2dMatrixBlocks("lnnaLB")
    def test2dMatrixBlocks_lnnaLA(self): self.check2dMatrixBlocks("lnnaLA")

    def test2dMatrixBlocks_lnnaAl(self): self.check2dMatrixBlocks("lnnaAl")
    def test2dMatrixBlocks_lnnaAL(self): self.check2dMatrixBlocks("lnnaAL")
    def test2dMatrixBlocks_lnnaAB(self): self.check2dMatrixBlocks("lnnaAB")
    def test2dMatrixBlocks_lnnaAA(self): self.check2dMatrixBlocks("lnnaAA")

    def test2dMatrixBlocks_lnnlLl(self): self.check2dMatrixBlocks("lnnlLl")
    def test2dMatrixBlocks_lnnlLL(self): self.check2dMatrixBlocks("lnnlLL")
    def test2dMatrixBlocks_lnnlLB(self): self.check2dMatrixBlocks("lnnlLB")
    def test2dMatrixBlocks_lnnlLA(self): self.check2dMatrixBlocks("lnnlLA")

    def test2dMatrixBlocks_lnnlAl(self): self.check2dMatrixBlocks("lnnlAl")
    def test2dMatrixBlocks_lnnlAL(self): self.check2dMatrixBlocks("lnnlAL")
    def test2dMatrixBlocks_lnnlAB(self): self.check2dMatrixBlocks("lnnlAB")
    def test2dMatrixBlocks_lnnlAA(self): self.check2dMatrixBlocks("lnnlAA")

    def test2dMatrixBlocks_lnnLLl(self): self.check2dMatrixBlocks("lnnLLl")
    def test2dMatrixBlocks_lnnLLL(self): self.check2dMatrixBlocks("lnnLLL")
    def test2dMatrixBlocks_lnnLLB(self): self.check2dMatrixBlocks("lnnLLB")
    def test2dMatrixBlocks_lnnLLA(self): self.check2dMatrixBlocks("lnnLLA")

    def test2dMatrixBlocks_lnnLAl(self): self.check2dMatrixBlocks("lnnLAl")
    def test2dMatrixBlocks_lnnLAL(self): self.check2dMatrixBlocks("lnnLAL")
    def test2dMatrixBlocks_lnnLAB(self): self.check2dMatrixBlocks("lnnLAB")
    def test2dMatrixBlocks_lnnLAA(self): self.check2dMatrixBlocks("lnnLAA")

    def test2dMatrixBlocks_lnnALl(self): self.check2dMatrixBlocks("lnnALl")
    def test2dMatrixBlocks_lnnALL(self): self.check2dMatrixBlocks("lnnALL")
    def test2dMatrixBlocks_lnnALB(self): self.check2dMatrixBlocks("lnnALB")
    def test2dMatrixBlocks_lnnALA(self): self.check2dMatrixBlocks("lnnALA")

    def test2dMatrixBlocks_lnnAAl(self): self.check2dMatrixBlocks("lnnAAl")
    def test2dMatrixBlocks_lnnAAL(self): self.check2dMatrixBlocks("lnnAAL")
    def test2dMatrixBlocks_lnnAAB(self): self.check2dMatrixBlocks("lnnAAB")
    def test2dMatrixBlocks_lnnAAA(self): self.check2dMatrixBlocks("lnnAAA")

    def test2dMatrixBlocks_lnNmLl(self): self.check2dMatrixBlocks("lnNmLl")
    def test2dMatrixBlocks_lnNmLL(self): self.check2dMatrixBlocks("lnNmLL")
    def test2dMatrixBlocks_lnNmLB(self): self.check2dMatrixBlocks("lnNmLB")
    def test2dMatrixBlocks_lnNmLA(self): self.check2dMatrixBlocks("lnNmLA")

    def test2dMatrixBlocks_lnNmAl(self): self.check2dMatrixBlocks("lnNmAl")
    def test2dMatrixBlocks_lnNmAL(self): self.check2dMatrixBlocks("lnNmAL")
    def test2dMatrixBlocks_lnNmAB(self): self.check2dMatrixBlocks("lnNmAB")
    def test2dMatrixBlocks_lnNmAA(self): self.check2dMatrixBlocks("lnNmAA")

    def test2dMatrixBlocks_lnNaLl(self): self.check2dMatrixBlocks("lnNaLl")
    def test2dMatrixBlocks_lnNaLL(self): self.check2dMatrixBlocks("lnNaLL")
    def test2dMatrixBlocks_lnNaLB(self): self.check2dMatrixBlocks("lnNaLB")
    def test2dMatrixBlocks_lnNaLA(self): self.check2dMatrixBlocks("lnNaLA")

    def test2dMatrixBlocks_lnNaAl(self): self.check2dMatrixBlocks("lnNaAl")
    def test2dMatrixBlocks_lnNaAL(self): self.check2dMatrixBlocks("lnNaAL")
    def test2dMatrixBlocks_lnNaAB(self): self.check2dMatrixBlocks("lnNaAB")
    def test2dMatrixBlocks_lnNaAA(self): self.check2dMatrixBlocks("lnNaAA")

    def test2dMatrixBlocks_lnNlLl(self): self.check2dMatrixBlocks("lnNlLl")
    def test2dMatrixBlocks_lnNlLL(self): self.check2dMatrixBlocks("lnNlLL")
    def test2dMatrixBlocks_lnNlLB(self): self.check2dMatrixBlocks("lnNlLB")
    def test2dMatrixBlocks_lnNlLA(self): self.check2dMatrixBlocks("lnNlLA")

    def test2dMatrixBlocks_lnNlAl(self): self.check2dMatrixBlocks("lnNlAl")
    def test2dMatrixBlocks_lnNlAL(self): self.check2dMatrixBlocks("lnNlAL")
    def test2dMatrixBlocks_lnNlAB(self): self.check2dMatrixBlocks("lnNlAB")
    def test2dMatrixBlocks_lnNlAA(self): self.check2dMatrixBlocks("lnNlAA")

    def test2dMatrixBlocks_lnNLLl(self): self.check2dMatrixBlocks("lnNLLl")
    def test2dMatrixBlocks_lnNLLL(self): self.check2dMatrixBlocks("lnNLLL")
    def test2dMatrixBlocks_lnNLLB(self): self.check2dMatrixBlocks("lnNLLB")
    def test2dMatrixBlocks_lnNLLA(self): self.check2dMatrixBlocks("lnNLLA")

    def test2dMatrixBlocks_lnNLAl(self): self.check2dMatrixBlocks("lnNLAl")
    def test2dMatrixBlocks_lnNLAL(self): self.check2dMatrixBlocks("lnNLAL")
    def test2dMatrixBlocks_lnNLAB(self): self.check2dMatrixBlocks("lnNLAB")
    def test2dMatrixBlocks_lnNLAA(self): self.check2dMatrixBlocks("lnNLAA")

    def test2dMatrixBlocks_lnNALl(self): self.check2dMatrixBlocks("lnNALl")
    def test2dMatrixBlocks_lnNALL(self): self.check2dMatrixBlocks("lnNALL")
    def test2dMatrixBlocks_lnNALB(self): self.check2dMatrixBlocks("lnNALB")
    def test2dMatrixBlocks_lnNALA(self): self.check2dMatrixBlocks("lnNALA")

    def test2dMatrixBlocks_lnNAAl(self): self.check2dMatrixBlocks("lnNAAl")
    def test2dMatrixBlocks_lnNAAL(self): self.check2dMatrixBlocks("lnNAAL")
    def test2dMatrixBlocks_lnNAAB(self): self.check2dMatrixBlocks("lnNAAB")
    def test2dMatrixBlocks_lnNAAA(self): self.check2dMatrixBlocks("lnNAAA")

    def test2dMatrixBlocks_lnlmLl(self): self.check2dMatrixBlocks("lnlmLl")
    def test2dMatrixBlocks_lnlmLL(self): self.check2dMatrixBlocks("lnlmLL")
    def test2dMatrixBlocks_lnlmLB(self): self.check2dMatrixBlocks("lnlmLB")
    def test2dMatrixBlocks_lnlmLA(self): self.check2dMatrixBlocks("lnlmLA")

    def test2dMatrixBlocks_lnlmAl(self): self.check2dMatrixBlocks("lnlmAl")
    def test2dMatrixBlocks_lnlmAL(self): self.check2dMatrixBlocks("lnlmAL")
    def test2dMatrixBlocks_lnlmAB(self): self.check2dMatrixBlocks("lnlmAB")
    def test2dMatrixBlocks_lnlmAA(self): self.check2dMatrixBlocks("lnlmAA")

    def test2dMatrixBlocks_lnlaLl(self): self.check2dMatrixBlocks("lnlaLl")
    def test2dMatrixBlocks_lnlaLL(self): self.check2dMatrixBlocks("lnlaLL")
    def test2dMatrixBlocks_lnlaLB(self): self.check2dMatrixBlocks("lnlaLB")
    def test2dMatrixBlocks_lnlaLA(self): self.check2dMatrixBlocks("lnlaLA")

    def test2dMatrixBlocks_lnlaAl(self): self.check2dMatrixBlocks("lnlaAl")
    def test2dMatrixBlocks_lnlaAL(self): self.check2dMatrixBlocks("lnlaAL")
    def test2dMatrixBlocks_lnlaAB(self): self.check2dMatrixBlocks("lnlaAB")
    def test2dMatrixBlocks_lnlaAA(self): self.check2dMatrixBlocks("lnlaAA")

    def test2dMatrixBlocks_lnllLl(self): self.check2dMatrixBlocks("lnllLl")
    def test2dMatrixBlocks_lnllLL(self): self.check2dMatrixBlocks("lnllLL")
    def test2dMatrixBlocks_lnllLB(self): self.check2dMatrixBlocks("lnllLB")
    def test2dMatrixBlocks_lnllLA(self): self.check2dMatrixBlocks("lnllLA")

    def test2dMatrixBlocks_lnllAl(self): self.check2dMatrixBlocks("lnllAl")
    def test2dMatrixBlocks_lnllAL(self): self.check2dMatrixBlocks("lnllAL")
    def test2dMatrixBlocks_lnllAB(self): self.check2dMatrixBlocks("lnllAB")
    def test2dMatrixBlocks_lnllAA(self): self.check2dMatrixBlocks("lnllAA")

    def test2dMatrixBlocks_lnlLLl(self): self.check2dMatrixBlocks("lnlLLl")
    def test2dMatrixBlocks_lnlLLL(self): self.check2dMatrixBlocks("lnlLLL")
    def test2dMatrixBlocks_lnlLLB(self): self.check2dMatrixBlocks("lnlLLB")
    def test2dMatrixBlocks_lnlLLA(self): self.check2dMatrixBlocks("lnlLLA")

    def test2dMatrixBlocks_lnlLAl(self): self.check2dMatrixBlocks("lnlLAl")
    def test2dMatrixBlocks_lnlLAL(self): self.check2dMatrixBlocks("lnlLAL")
    def test2dMatrixBlocks_lnlLAB(self): self.check2dMatrixBlocks("lnlLAB")
    def test2dMatrixBlocks_lnlLAA(self): self.check2dMatrixBlocks("lnlLAA")

    def test2dMatrixBlocks_lnlALl(self): self.check2dMatrixBlocks("lnlALl")
    def test2dMatrixBlocks_lnlALL(self): self.check2dMatrixBlocks("lnlALL")
    def test2dMatrixBlocks_lnlALB(self): self.check2dMatrixBlocks("lnlALB")
    def test2dMatrixBlocks_lnlALA(self): self.check2dMatrixBlocks("lnlALA")

    def test2dMatrixBlocks_lnlAAl(self): self.check2dMatrixBlocks("lnlAAl")
    def test2dMatrixBlocks_lnlAAL(self): self.check2dMatrixBlocks("lnlAAL")
    def test2dMatrixBlocks_lnlAAB(self): self.check2dMatrixBlocks("lnlAAB")
    def test2dMatrixBlocks_lnlAAA(self): self.check2dMatrixBlocks("lnlAAA")

    def test2dMatrixBlocks_lNtsLl(self): self.check2dMatrixBlocks("lNtsLl")
    def test2dMatrixBlocks_lNtsLL(self): self.check2dMatrixBlocks("lNtsLL")
    def test2dMatrixBlocks_lNtsLB(self): self.check2dMatrixBlocks("lNtsLB")
    def test2dMatrixBlocks_lNtsLA(self): self.check2dMatrixBlocks("lNtsLA")

    def test2dMatrixBlocks_lNtsAl(self): self.check2dMatrixBlocks("lNtsAl")
    def test2dMatrixBlocks_lNtsAL(self): self.check2dMatrixBlocks("lNtsAL")
    def test2dMatrixBlocks_lNtsAB(self): self.check2dMatrixBlocks("lNtsAB")
    def test2dMatrixBlocks_lNtsAA(self): self.check2dMatrixBlocks("lNtsAA")

    def test2dMatrixBlocks_lNtmLl(self): self.check2dMatrixBlocks("lNtmLl")
    def test2dMatrixBlocks_lNtmLL(self): self.check2dMatrixBlocks("lNtmLL")
    def test2dMatrixBlocks_lNtmLB(self): self.check2dMatrixBlocks("lNtmLB")
    def test2dMatrixBlocks_lNtmLA(self): self.check2dMatrixBlocks("lNtmLA")

    def test2dMatrixBlocks_lNtmAl(self): self.check2dMatrixBlocks("lNtmAl")
    def test2dMatrixBlocks_lNtmAL(self): self.check2dMatrixBlocks("lNtmAL")
    def test2dMatrixBlocks_lNtmAB(self): self.check2dMatrixBlocks("lNtmAB")
    def test2dMatrixBlocks_lNtmAA(self): self.check2dMatrixBlocks("lNtmAA")

    def test2dMatrixBlocks_lNtaLl(self): self.check2dMatrixBlocks("lNtaLl")
    def test2dMatrixBlocks_lNtaLL(self): self.check2dMatrixBlocks("lNtaLL")
    def test2dMatrixBlocks_lNtaLB(self): self.check2dMatrixBlocks("lNtaLB")
    def test2dMatrixBlocks_lNtaLA(self): self.check2dMatrixBlocks("lNtaLA")

    def test2dMatrixBlocks_lNtaAl(self): self.check2dMatrixBlocks("lNtaAl")
    def test2dMatrixBlocks_lNtaAL(self): self.check2dMatrixBlocks("lNtaAL")
    def test2dMatrixBlocks_lNtaAB(self): self.check2dMatrixBlocks("lNtaAB")
    def test2dMatrixBlocks_lNtaAA(self): self.check2dMatrixBlocks("lNtaAA")

    def test2dMatrixBlocks_lNtlLl(self): self.check2dMatrixBlocks("lNtlLl")
    def test2dMatrixBlocks_lNtlLL(self): self.check2dMatrixBlocks("lNtlLL")
    def test2dMatrixBlocks_lNtlLB(self): self.check2dMatrixBlocks("lNtlLB")
    def test2dMatrixBlocks_lNtlLA(self): self.check2dMatrixBlocks("lNtlLA")

    def test2dMatrixBlocks_lNtlAl(self): self.check2dMatrixBlocks("lNtlAl")
    def test2dMatrixBlocks_lNtlAL(self): self.check2dMatrixBlocks("lNtlAL")
    def test2dMatrixBlocks_lNtlAB(self): self.check2dMatrixBlocks("lNtlAB")
    def test2dMatrixBlocks_lNtlAA(self): self.check2dMatrixBlocks("lNtlAA")

    def test2dMatrixBlocks_lNtLLl(self): self.check2dMatrixBlocks("lNtLLl")
    def test2dMatrixBlocks_lNtLLL(self): self.check2dMatrixBlocks("lNtLLL")
    def test2dMatrixBlocks_lNtLLB(self): self.check2dMatrixBlocks("lNtLLB")
    def test2dMatrixBlocks_lNtLLA(self): self.check2dMatrixBlocks("lNtLLA")

    def test2dMatrixBlocks_lNtLAl(self): self.check2dMatrixBlocks("lNtLAl")
    def test2dMatrixBlocks_lNtLAL(self): self.check2dMatrixBlocks("lNtLAL")
    def test2dMatrixBlocks_lNtLAB(self): self.check2dMatrixBlocks("lNtLAB")
    def test2dMatrixBlocks_lNtLAA(self): self.check2dMatrixBlocks("lNtLAA")

    def test2dMatrixBlocks_lNtALl(self): self.check2dMatrixBlocks("lNtALl")
    def test2dMatrixBlocks_lNtALL(self): self.check2dMatrixBlocks("lNtALL")
    def test2dMatrixBlocks_lNtALB(self): self.check2dMatrixBlocks("lNtALB")
    def test2dMatrixBlocks_lNtALA(self): self.check2dMatrixBlocks("lNtALA")

    def test2dMatrixBlocks_lNtAAl(self): self.check2dMatrixBlocks("lNtAAl")
    def test2dMatrixBlocks_lNtAAL(self): self.check2dMatrixBlocks("lNtAAL")
    def test2dMatrixBlocks_lNtAAB(self): self.check2dMatrixBlocks("lNtAAB")
    def test2dMatrixBlocks_lNtAAA(self): self.check2dMatrixBlocks("lNtAAA")

    def test2dMatrixBlocks_lNnsLl(self): self.check2dMatrixBlocks("lNnsLl")
    def test2dMatrixBlocks_lNnsLL(self): self.check2dMatrixBlocks("lNnsLL")
    def test2dMatrixBlocks_lNnsLB(self): self.check2dMatrixBlocks("lNnsLB")
    def test2dMatrixBlocks_lNnsLA(self): self.check2dMatrixBlocks("lNnsLA")

    def test2dMatrixBlocks_lNnsAl(self): self.check2dMatrixBlocks("lNnsAl")
    def test2dMatrixBlocks_lNnsAL(self): self.check2dMatrixBlocks("lNnsAL")
    def test2dMatrixBlocks_lNnsAB(self): self.check2dMatrixBlocks("lNnsAB")
    def test2dMatrixBlocks_lNnsAA(self): self.check2dMatrixBlocks("lNnsAA")

    def test2dMatrixBlocks_lNnmLl(self): self.check2dMatrixBlocks("lNnmLl")
    def test2dMatrixBlocks_lNnmLL(self): self.check2dMatrixBlocks("lNnmLL")
    def test2dMatrixBlocks_lNnmLB(self): self.check2dMatrixBlocks("lNnmLB")
    def test2dMatrixBlocks_lNnmLA(self): self.check2dMatrixBlocks("lNnmLA")

    def test2dMatrixBlocks_lNnmAl(self): self.check2dMatrixBlocks("lNnmAl")
    def test2dMatrixBlocks_lNnmAL(self): self.check2dMatrixBlocks("lNnmAL")
    def test2dMatrixBlocks_lNnmAB(self): self.check2dMatrixBlocks("lNnmAB")
    def test2dMatrixBlocks_lNnmAA(self): self.check2dMatrixBlocks("lNnmAA")

    def test2dMatrixBlocks_lNnaLl(self): self.check2dMatrixBlocks("lNnaLl")
    def test2dMatrixBlocks_lNnaLL(self): self.check2dMatrixBlocks("lNnaLL")
    def test2dMatrixBlocks_lNnaLB(self): self.check2dMatrixBlocks("lNnaLB")
    def test2dMatrixBlocks_lNnaLA(self): self.check2dMatrixBlocks("lNnaLA")

    def test2dMatrixBlocks_lNnaAl(self): self.check2dMatrixBlocks("lNnaAl")
    def test2dMatrixBlocks_lNnaAL(self): self.check2dMatrixBlocks("lNnaAL")
    def test2dMatrixBlocks_lNnaAB(self): self.check2dMatrixBlocks("lNnaAB")
    def test2dMatrixBlocks_lNnaAA(self): self.check2dMatrixBlocks("lNnaAA")

    def test2dMatrixBlocks_lNnlLl(self): self.check2dMatrixBlocks("lNnlLl")
    def test2dMatrixBlocks_lNnlLL(self): self.check2dMatrixBlocks("lNnlLL")
    def test2dMatrixBlocks_lNnlLB(self): self.check2dMatrixBlocks("lNnlLB")
    def test2dMatrixBlocks_lNnlLA(self): self.check2dMatrixBlocks("lNnlLA")

    def test2dMatrixBlocks_lNnlAl(self): self.check2dMatrixBlocks("lNnlAl")
    def test2dMatrixBlocks_lNnlAL(self): self.check2dMatrixBlocks("lNnlAL")
    def test2dMatrixBlocks_lNnlAB(self): self.check2dMatrixBlocks("lNnlAB")
    def test2dMatrixBlocks_lNnlAA(self): self.check2dMatrixBlocks("lNnlAA")

    def test2dMatrixBlocks_lNnLLl(self): self.check2dMatrixBlocks("lNnLLl")
    def test2dMatrixBlocks_lNnLLL(self): self.check2dMatrixBlocks("lNnLLL")
    def test2dMatrixBlocks_lNnLLB(self): self.check2dMatrixBlocks("lNnLLB")
    def test2dMatrixBlocks_lNnLLA(self): self.check2dMatrixBlocks("lNnLLA")

    def test2dMatrixBlocks_lNnLAl(self): self.check2dMatrixBlocks("lNnLAl")
    def test2dMatrixBlocks_lNnLAL(self): self.check2dMatrixBlocks("lNnLAL")
    def test2dMatrixBlocks_lNnLAB(self): self.check2dMatrixBlocks("lNnLAB")
    def test2dMatrixBlocks_lNnLAA(self): self.check2dMatrixBlocks("lNnLAA")

    def test2dMatrixBlocks_lNnALl(self): self.check2dMatrixBlocks("lNnALl")
    def test2dMatrixBlocks_lNnALL(self): self.check2dMatrixBlocks("lNnALL")
    def test2dMatrixBlocks_lNnALB(self): self.check2dMatrixBlocks("lNnALB")
    def test2dMatrixBlocks_lNnALA(self): self.check2dMatrixBlocks("lNnALA")

    def test2dMatrixBlocks_lNnAAl(self): self.check2dMatrixBlocks("lNnAAl")
    def test2dMatrixBlocks_lNnAAL(self): self.check2dMatrixBlocks("lNnAAL")
    def test2dMatrixBlocks_lNnAAB(self): self.check2dMatrixBlocks("lNnAAB")
    def test2dMatrixBlocks_lNnAAA(self): self.check2dMatrixBlocks("lNnAAA")

    def test2dMatrixBlocks_lNNsLl(self): self.check2dMatrixBlocks("lNNsLl")
    def test2dMatrixBlocks_lNNsLL(self): self.check2dMatrixBlocks("lNNsLL")
    def test2dMatrixBlocks_lNNsLB(self): self.check2dMatrixBlocks("lNNsLB")
    def test2dMatrixBlocks_lNNsLA(self): self.check2dMatrixBlocks("lNNsLA")

    def test2dMatrixBlocks_lNNsAl(self): self.check2dMatrixBlocks("lNNsAl")
    def test2dMatrixBlocks_lNNsAL(self): self.check2dMatrixBlocks("lNNsAL")
    def test2dMatrixBlocks_lNNsAB(self): self.check2dMatrixBlocks("lNNsAB")
    def test2dMatrixBlocks_lNNsAA(self): self.check2dMatrixBlocks("lNNsAA")

    def test2dMatrixBlocks_lNNmLl(self): self.check2dMatrixBlocks("lNNmLl")
    def test2dMatrixBlocks_lNNmLL(self): self.check2dMatrixBlocks("lNNmLL")
    def test2dMatrixBlocks_lNNmLB(self): self.check2dMatrixBlocks("lNNmLB")
    def test2dMatrixBlocks_lNNmLA(self): self.check2dMatrixBlocks("lNNmLA")

    def test2dMatrixBlocks_lNNmAl(self): self.check2dMatrixBlocks("lNNmAl")
    def test2dMatrixBlocks_lNNmAL(self): self.check2dMatrixBlocks("lNNmAL")
    def test2dMatrixBlocks_lNNmAB(self): self.check2dMatrixBlocks("lNNmAB")
    def test2dMatrixBlocks_lNNmAA(self): self.check2dMatrixBlocks("lNNmAA")

    def test2dMatrixBlocks_lNNaLl(self): self.check2dMatrixBlocks("lNNaLl")
    def test2dMatrixBlocks_lNNaLL(self): self.check2dMatrixBlocks("lNNaLL")
    def test2dMatrixBlocks_lNNaLB(self): self.check2dMatrixBlocks("lNNaLB")
    def test2dMatrixBlocks_lNNaLA(self): self.check2dMatrixBlocks("lNNaLA")

    def test2dMatrixBlocks_lNNaAl(self): self.check2dMatrixBlocks("lNNaAl")
    def test2dMatrixBlocks_lNNaAL(self): self.check2dMatrixBlocks("lNNaAL")
    def test2dMatrixBlocks_lNNaAB(self): self.check2dMatrixBlocks("lNNaAB")
    def test2dMatrixBlocks_lNNaAA(self): self.check2dMatrixBlocks("lNNaAA")

    def test2dMatrixBlocks_lNNlLl(self): self.check2dMatrixBlocks("lNNlLl")
    def test2dMatrixBlocks_lNNlLL(self): self.check2dMatrixBlocks("lNNlLL")
    def test2dMatrixBlocks_lNNlLB(self): self.check2dMatrixBlocks("lNNlLB")
    def test2dMatrixBlocks_lNNlLA(self): self.check2dMatrixBlocks("lNNlLA")

    def test2dMatrixBlocks_lNNlAl(self): self.check2dMatrixBlocks("lNNlAl")
    def test2dMatrixBlocks_lNNlAL(self): self.check2dMatrixBlocks("lNNlAL")
    def test2dMatrixBlocks_lNNlAB(self): self.check2dMatrixBlocks("lNNlAB")
    def test2dMatrixBlocks_lNNlAA(self): self.check2dMatrixBlocks("lNNlAA")

    def test2dMatrixBlocks_lNNLLl(self): self.check2dMatrixBlocks("lNNLLl")
    def test2dMatrixBlocks_lNNLLL(self): self.check2dMatrixBlocks("lNNLLL")
    def test2dMatrixBlocks_lNNLLB(self): self.check2dMatrixBlocks("lNNLLB")
    def test2dMatrixBlocks_lNNLLA(self): self.check2dMatrixBlocks("lNNLLA")

    def test2dMatrixBlocks_lNNLAl(self): self.check2dMatrixBlocks("lNNLAl")
    def test2dMatrixBlocks_lNNLAL(self): self.check2dMatrixBlocks("lNNLAL")
    def test2dMatrixBlocks_lNNLAB(self): self.check2dMatrixBlocks("lNNLAB")
    def test2dMatrixBlocks_lNNLAA(self): self.check2dMatrixBlocks("lNNLAA")

    def test2dMatrixBlocks_lNNALl(self): self.check2dMatrixBlocks("lNNALl")
    def test2dMatrixBlocks_lNNALL(self): self.check2dMatrixBlocks("lNNALL")
    def test2dMatrixBlocks_lNNALB(self): self.check2dMatrixBlocks("lNNALB")
    def test2dMatrixBlocks_lNNALA(self): self.check2dMatrixBlocks("lNNALA")

    def test2dMatrixBlocks_lNNAAl(self): self.check2dMatrixBlocks("lNNAAl")
    def test2dMatrixBlocks_lNNAAL(self): self.check2dMatrixBlocks("lNNAAL")
    def test2dMatrixBlocks_lNNAAB(self): self.check2dMatrixBlocks("lNNAAB")
    def test2dMatrixBlocks_lNNAAA(self): self.check2dMatrixBlocks("lNNAAA")

    def test2dMatrixBlocks_lNlsLl(self): self.check2dMatrixBlocks("lNlsLl")
    def test2dMatrixBlocks_lNlsLL(self): self.check2dMatrixBlocks("lNlsLL")
    def test2dMatrixBlocks_lNlsLB(self): self.check2dMatrixBlocks("lNlsLB")
    def test2dMatrixBlocks_lNlsLA(self): self.check2dMatrixBlocks("lNlsLA")

    def test2dMatrixBlocks_lNlsAl(self): self.check2dMatrixBlocks("lNlsAl")
    def test2dMatrixBlocks_lNlsAL(self): self.check2dMatrixBlocks("lNlsAL")
    def test2dMatrixBlocks_lNlsAB(self): self.check2dMatrixBlocks("lNlsAB")
    def test2dMatrixBlocks_lNlsAA(self): self.check2dMatrixBlocks("lNlsAA")

    def test2dMatrixBlocks_lNlmLl(self): self.check2dMatrixBlocks("lNlmLl")
    def test2dMatrixBlocks_lNlmLL(self): self.check2dMatrixBlocks("lNlmLL")
    def test2dMatrixBlocks_lNlmLB(self): self.check2dMatrixBlocks("lNlmLB")
    def test2dMatrixBlocks_lNlmLA(self): self.check2dMatrixBlocks("lNlmLA")

    def test2dMatrixBlocks_lNlmAl(self): self.check2dMatrixBlocks("lNlmAl")
    def test2dMatrixBlocks_lNlmAL(self): self.check2dMatrixBlocks("lNlmAL")
    def test2dMatrixBlocks_lNlmAB(self): self.check2dMatrixBlocks("lNlmAB")
    def test2dMatrixBlocks_lNlmAA(self): self.check2dMatrixBlocks("lNlmAA")

    def test2dMatrixBlocks_lNlaLl(self): self.check2dMatrixBlocks("lNlaLl")
    def test2dMatrixBlocks_lNlaLL(self): self.check2dMatrixBlocks("lNlaLL")
    def test2dMatrixBlocks_lNlaLB(self): self.check2dMatrixBlocks("lNlaLB")
    def test2dMatrixBlocks_lNlaLA(self): self.check2dMatrixBlocks("lNlaLA")

    def test2dMatrixBlocks_lNlaAl(self): self.check2dMatrixBlocks("lNlaAl")
    def test2dMatrixBlocks_lNlaAL(self): self.check2dMatrixBlocks("lNlaAL")
    def test2dMatrixBlocks_lNlaAB(self): self.check2dMatrixBlocks("lNlaAB")
    def test2dMatrixBlocks_lNlaAA(self): self.check2dMatrixBlocks("lNlaAA")

    def test2dMatrixBlocks_lNllLl(self): self.check2dMatrixBlocks("lNllLl")
    def test2dMatrixBlocks_lNllLL(self): self.check2dMatrixBlocks("lNllLL")
    def test2dMatrixBlocks_lNllLB(self): self.check2dMatrixBlocks("lNllLB")
    def test2dMatrixBlocks_lNllLA(self): self.check2dMatrixBlocks("lNllLA")

    def test2dMatrixBlocks_lNllAl(self): self.check2dMatrixBlocks("lNllAl")
    def test2dMatrixBlocks_lNllAL(self): self.check2dMatrixBlocks("lNllAL")
    def test2dMatrixBlocks_lNllAB(self): self.check2dMatrixBlocks("lNllAB")
    def test2dMatrixBlocks_lNllAA(self): self.check2dMatrixBlocks("lNllAA")

    def test2dMatrixBlocks_lNlLLl(self): self.check2dMatrixBlocks("lNlLLl")
    def test2dMatrixBlocks_lNlLLL(self): self.check2dMatrixBlocks("lNlLLL")
    def test2dMatrixBlocks_lNlLLB(self): self.check2dMatrixBlocks("lNlLLB")
    def test2dMatrixBlocks_lNlLLA(self): self.check2dMatrixBlocks("lNlLLA")

    def test2dMatrixBlocks_lNlLAl(self): self.check2dMatrixBlocks("lNlLAl")
    def test2dMatrixBlocks_lNlLAL(self): self.check2dMatrixBlocks("lNlLAL")
    def test2dMatrixBlocks_lNlLAB(self): self.check2dMatrixBlocks("lNlLAB")
    def test2dMatrixBlocks_lNlLAA(self): self.check2dMatrixBlocks("lNlLAA")

    def test2dMatrixBlocks_lNlALl(self): self.check2dMatrixBlocks("lNlALl")
    def test2dMatrixBlocks_lNlALL(self): self.check2dMatrixBlocks("lNlALL")
    def test2dMatrixBlocks_lNlALB(self): self.check2dMatrixBlocks("lNlALB")
    def test2dMatrixBlocks_lNlALA(self): self.check2dMatrixBlocks("lNlALA")

    def test2dMatrixBlocks_lNlAAl(self): self.check2dMatrixBlocks("lNlAAl")
    def test2dMatrixBlocks_lNlAAL(self): self.check2dMatrixBlocks("lNlAAL")
    def test2dMatrixBlocks_lNlAAB(self): self.check2dMatrixBlocks("lNlAAB")
    def test2dMatrixBlocks_lNlAAA(self): self.check2dMatrixBlocks("lNlAAA")

    def test2dMatrixBlocks_lltsLl(self): self.check2dMatrixBlocks("lltsLl")
    def test2dMatrixBlocks_lltsLL(self): self.check2dMatrixBlocks("lltsLL")
    def test2dMatrixBlocks_lltsLB(self): self.check2dMatrixBlocks("lltsLB")
    def test2dMatrixBlocks_lltsLA(self): self.check2dMatrixBlocks("lltsLA")

    def test2dMatrixBlocks_lltsAl(self): self.check2dMatrixBlocks("lltsAl")
    def test2dMatrixBlocks_lltsAL(self): self.check2dMatrixBlocks("lltsAL")
    def test2dMatrixBlocks_lltsAB(self): self.check2dMatrixBlocks("lltsAB")
    def test2dMatrixBlocks_lltsAA(self): self.check2dMatrixBlocks("lltsAA")

    def test2dMatrixBlocks_lltmLl(self): self.check2dMatrixBlocks("lltmLl")
    def test2dMatrixBlocks_lltmLL(self): self.check2dMatrixBlocks("lltmLL")
    def test2dMatrixBlocks_lltmLB(self): self.check2dMatrixBlocks("lltmLB")
    def test2dMatrixBlocks_lltmLA(self): self.check2dMatrixBlocks("lltmLA")

    def test2dMatrixBlocks_lltmAl(self): self.check2dMatrixBlocks("lltmAl")
    def test2dMatrixBlocks_lltmAL(self): self.check2dMatrixBlocks("lltmAL")
    def test2dMatrixBlocks_lltmAB(self): self.check2dMatrixBlocks("lltmAB")
    def test2dMatrixBlocks_lltmAA(self): self.check2dMatrixBlocks("lltmAA")

    def test2dMatrixBlocks_lltaLl(self): self.check2dMatrixBlocks("lltaLl")
    def test2dMatrixBlocks_lltaLL(self): self.check2dMatrixBlocks("lltaLL")
    def test2dMatrixBlocks_lltaLB(self): self.check2dMatrixBlocks("lltaLB")
    def test2dMatrixBlocks_lltaLA(self): self.check2dMatrixBlocks("lltaLA")

    def test2dMatrixBlocks_lltaAl(self): self.check2dMatrixBlocks("lltaAl")
    def test2dMatrixBlocks_lltaAL(self): self.check2dMatrixBlocks("lltaAL")
    def test2dMatrixBlocks_lltaAB(self): self.check2dMatrixBlocks("lltaAB")
    def test2dMatrixBlocks_lltaAA(self): self.check2dMatrixBlocks("lltaAA")

    def test2dMatrixBlocks_lltlLl(self): self.check2dMatrixBlocks("lltlLl")
    def test2dMatrixBlocks_lltlLL(self): self.check2dMatrixBlocks("lltlLL")
    def test2dMatrixBlocks_lltlLB(self): self.check2dMatrixBlocks("lltlLB")
    def test2dMatrixBlocks_lltlLA(self): self.check2dMatrixBlocks("lltlLA")

    def test2dMatrixBlocks_lltlAl(self): self.check2dMatrixBlocks("lltlAl")
    def test2dMatrixBlocks_lltlAL(self): self.check2dMatrixBlocks("lltlAL")
    def test2dMatrixBlocks_lltlAB(self): self.check2dMatrixBlocks("lltlAB")
    def test2dMatrixBlocks_lltlAA(self): self.check2dMatrixBlocks("lltlAA")

    def test2dMatrixBlocks_lltLLl(self): self.check2dMatrixBlocks("lltLLl")
    def test2dMatrixBlocks_lltLLL(self): self.check2dMatrixBlocks("lltLLL")
    def test2dMatrixBlocks_lltLLB(self): self.check2dMatrixBlocks("lltLLB")
    def test2dMatrixBlocks_lltLLA(self): self.check2dMatrixBlocks("lltLLA")

    def test2dMatrixBlocks_lltLAl(self): self.check2dMatrixBlocks("lltLAl")
    def test2dMatrixBlocks_lltLAL(self): self.check2dMatrixBlocks("lltLAL")
    def test2dMatrixBlocks_lltLAB(self): self.check2dMatrixBlocks("lltLAB")
    def test2dMatrixBlocks_lltLAA(self): self.check2dMatrixBlocks("lltLAA")

    def test2dMatrixBlocks_lltALl(self): self.check2dMatrixBlocks("lltALl")
    def test2dMatrixBlocks_lltALL(self): self.check2dMatrixBlocks("lltALL")
    def test2dMatrixBlocks_lltALB(self): self.check2dMatrixBlocks("lltALB")
    def test2dMatrixBlocks_lltALA(self): self.check2dMatrixBlocks("lltALA")

    def test2dMatrixBlocks_lltAAl(self): self.check2dMatrixBlocks("lltAAl")
    def test2dMatrixBlocks_lltAAL(self): self.check2dMatrixBlocks("lltAAL")
    def test2dMatrixBlocks_lltAAB(self): self.check2dMatrixBlocks("lltAAB")
    def test2dMatrixBlocks_lltAAA(self): self.check2dMatrixBlocks("lltAAA")

    def test2dMatrixBlocks_llnsLl(self): self.check2dMatrixBlocks("llnsLl")
    def test2dMatrixBlocks_llnsLL(self): self.check2dMatrixBlocks("llnsLL")
    def test2dMatrixBlocks_llnsLB(self): self.check2dMatrixBlocks("llnsLB")
    def test2dMatrixBlocks_llnsLA(self): self.check2dMatrixBlocks("llnsLA")

    def test2dMatrixBlocks_llnsAl(self): self.check2dMatrixBlocks("llnsAl")
    def test2dMatrixBlocks_llnsAL(self): self.check2dMatrixBlocks("llnsAL")
    def test2dMatrixBlocks_llnsAB(self): self.check2dMatrixBlocks("llnsAB")
    def test2dMatrixBlocks_llnsAA(self): self.check2dMatrixBlocks("llnsAA")

    def test2dMatrixBlocks_llnmLl(self): self.check2dMatrixBlocks("llnmLl")
    def test2dMatrixBlocks_llnmLL(self): self.check2dMatrixBlocks("llnmLL")
    def test2dMatrixBlocks_llnmLB(self): self.check2dMatrixBlocks("llnmLB")
    def test2dMatrixBlocks_llnmLA(self): self.check2dMatrixBlocks("llnmLA")

    def test2dMatrixBlocks_llnmAl(self): self.check2dMatrixBlocks("llnmAl")
    def test2dMatrixBlocks_llnmAL(self): self.check2dMatrixBlocks("llnmAL")
    def test2dMatrixBlocks_llnmAB(self): self.check2dMatrixBlocks("llnmAB")
    def test2dMatrixBlocks_llnmAA(self): self.check2dMatrixBlocks("llnmAA")

    def test2dMatrixBlocks_llnaLl(self): self.check2dMatrixBlocks("llnaLl")
    def test2dMatrixBlocks_llnaLL(self): self.check2dMatrixBlocks("llnaLL")
    def test2dMatrixBlocks_llnaLB(self): self.check2dMatrixBlocks("llnaLB")
    def test2dMatrixBlocks_llnaLA(self): self.check2dMatrixBlocks("llnaLA")

    def test2dMatrixBlocks_llnaAl(self): self.check2dMatrixBlocks("llnaAl")
    def test2dMatrixBlocks_llnaAL(self): self.check2dMatrixBlocks("llnaAL")
    def test2dMatrixBlocks_llnaAB(self): self.check2dMatrixBlocks("llnaAB")
    def test2dMatrixBlocks_llnaAA(self): self.check2dMatrixBlocks("llnaAA")

    def test2dMatrixBlocks_llnlLl(self): self.check2dMatrixBlocks("llnlLl")
    def test2dMatrixBlocks_llnlLL(self): self.check2dMatrixBlocks("llnlLL")
    def test2dMatrixBlocks_llnlLB(self): self.check2dMatrixBlocks("llnlLB")
    def test2dMatrixBlocks_llnlLA(self): self.check2dMatrixBlocks("llnlLA")

    def test2dMatrixBlocks_llnlAl(self): self.check2dMatrixBlocks("llnlAl")
    def test2dMatrixBlocks_llnlAL(self): self.check2dMatrixBlocks("llnlAL")
    def test2dMatrixBlocks_llnlAB(self): self.check2dMatrixBlocks("llnlAB")
    def test2dMatrixBlocks_llnlAA(self): self.check2dMatrixBlocks("llnlAA")

    def test2dMatrixBlocks_llnLLl(self): self.check2dMatrixBlocks("llnLLl")
    def test2dMatrixBlocks_llnLLL(self): self.check2dMatrixBlocks("llnLLL")
    def test2dMatrixBlocks_llnLLB(self): self.check2dMatrixBlocks("llnLLB")
    def test2dMatrixBlocks_llnLLA(self): self.check2dMatrixBlocks("llnLLA")

    def test2dMatrixBlocks_llnLAl(self): self.check2dMatrixBlocks("llnLAl")
    def test2dMatrixBlocks_llnLAL(self): self.check2dMatrixBlocks("llnLAL")
    def test2dMatrixBlocks_llnLAB(self): self.check2dMatrixBlocks("llnLAB")
    def test2dMatrixBlocks_llnLAA(self): self.check2dMatrixBlocks("llnLAA")

    def test2dMatrixBlocks_llnALl(self): self.check2dMatrixBlocks("llnALl")
    def test2dMatrixBlocks_llnALL(self): self.check2dMatrixBlocks("llnALL")
    def test2dMatrixBlocks_llnALB(self): self.check2dMatrixBlocks("llnALB")
    def test2dMatrixBlocks_llnALA(self): self.check2dMatrixBlocks("llnALA")

    def test2dMatrixBlocks_llnAAl(self): self.check2dMatrixBlocks("llnAAl")
    def test2dMatrixBlocks_llnAAL(self): self.check2dMatrixBlocks("llnAAL")
    def test2dMatrixBlocks_llnAAB(self): self.check2dMatrixBlocks("llnAAB")
    def test2dMatrixBlocks_llnAAA(self): self.check2dMatrixBlocks("llnAAA")

    def test2dMatrixBlocks_llNsLl(self): self.check2dMatrixBlocks("llNsLl")
    def test2dMatrixBlocks_llNsLL(self): self.check2dMatrixBlocks("llNsLL")
    def test2dMatrixBlocks_llNsLB(self): self.check2dMatrixBlocks("llNsLB")
    def test2dMatrixBlocks_llNsLA(self): self.check2dMatrixBlocks("llNsLA")

    def test2dMatrixBlocks_llNsAl(self): self.check2dMatrixBlocks("llNsAl")
    def test2dMatrixBlocks_llNsAL(self): self.check2dMatrixBlocks("llNsAL")
    def test2dMatrixBlocks_llNsAB(self): self.check2dMatrixBlocks("llNsAB")
    def test2dMatrixBlocks_llNsAA(self): self.check2dMatrixBlocks("llNsAA")

    def test2dMatrixBlocks_llNmLl(self): self.check2dMatrixBlocks("llNmLl")
    def test2dMatrixBlocks_llNmLL(self): self.check2dMatrixBlocks("llNmLL")
    def test2dMatrixBlocks_llNmLB(self): self.check2dMatrixBlocks("llNmLB")
    def test2dMatrixBlocks_llNmLA(self): self.check2dMatrixBlocks("llNmLA")

    def test2dMatrixBlocks_llNmAl(self): self.check2dMatrixBlocks("llNmAl")
    def test2dMatrixBlocks_llNmAL(self): self.check2dMatrixBlocks("llNmAL")
    def test2dMatrixBlocks_llNmAB(self): self.check2dMatrixBlocks("llNmAB")
    def test2dMatrixBlocks_llNmAA(self): self.check2dMatrixBlocks("llNmAA")

    def test2dMatrixBlocks_llNaLl(self): self.check2dMatrixBlocks("llNaLl")
    def test2dMatrixBlocks_llNaLL(self): self.check2dMatrixBlocks("llNaLL")
    def test2dMatrixBlocks_llNaLB(self): self.check2dMatrixBlocks("llNaLB")
    def test2dMatrixBlocks_llNaLA(self): self.check2dMatrixBlocks("llNaLA")

    def test2dMatrixBlocks_llNaAl(self): self.check2dMatrixBlocks("llNaAl")
    def test2dMatrixBlocks_llNaAL(self): self.check2dMatrixBlocks("llNaAL")
    def test2dMatrixBlocks_llNaAB(self): self.check2dMatrixBlocks("llNaAB")
    def test2dMatrixBlocks_llNaAA(self): self.check2dMatrixBlocks("llNaAA")

    def test2dMatrixBlocks_llNlLl(self): self.check2dMatrixBlocks("llNlLl")
    def test2dMatrixBlocks_llNlLL(self): self.check2dMatrixBlocks("llNlLL")
    def test2dMatrixBlocks_llNlLB(self): self.check2dMatrixBlocks("llNlLB")
    def test2dMatrixBlocks_llNlLA(self): self.check2dMatrixBlocks("llNlLA")

    def test2dMatrixBlocks_llNlAl(self): self.check2dMatrixBlocks("llNlAl")
    def test2dMatrixBlocks_llNlAL(self): self.check2dMatrixBlocks("llNlAL")
    def test2dMatrixBlocks_llNlAB(self): self.check2dMatrixBlocks("llNlAB")
    def test2dMatrixBlocks_llNlAA(self): self.check2dMatrixBlocks("llNlAA")

    def test2dMatrixBlocks_llNLLl(self): self.check2dMatrixBlocks("llNLLl")
    def test2dMatrixBlocks_llNLLL(self): self.check2dMatrixBlocks("llNLLL")
    def test2dMatrixBlocks_llNLLB(self): self.check2dMatrixBlocks("llNLLB")
    def test2dMatrixBlocks_llNLLA(self): self.check2dMatrixBlocks("llNLLA")

    def test2dMatrixBlocks_llNLAl(self): self.check2dMatrixBlocks("llNLAl")
    def test2dMatrixBlocks_llNLAL(self): self.check2dMatrixBlocks("llNLAL")
    def test2dMatrixBlocks_llNLAB(self): self.check2dMatrixBlocks("llNLAB")
    def test2dMatrixBlocks_llNLAA(self): self.check2dMatrixBlocks("llNLAA")

    def test2dMatrixBlocks_llNALl(self): self.check2dMatrixBlocks("llNALl")
    def test2dMatrixBlocks_llNALL(self): self.check2dMatrixBlocks("llNALL")
    def test2dMatrixBlocks_llNALB(self): self.check2dMatrixBlocks("llNALB")
    def test2dMatrixBlocks_llNALA(self): self.check2dMatrixBlocks("llNALA")

    def test2dMatrixBlocks_llNAAl(self): self.check2dMatrixBlocks("llNAAl")
    def test2dMatrixBlocks_llNAAL(self): self.check2dMatrixBlocks("llNAAL")
    def test2dMatrixBlocks_llNAAB(self): self.check2dMatrixBlocks("llNAAB")
    def test2dMatrixBlocks_llNAAA(self): self.check2dMatrixBlocks("llNAAA")

    def test2dMatrixBlocks_lllsLl(self): self.check2dMatrixBlocks("lllsLl")
    def test2dMatrixBlocks_lllsLL(self): self.check2dMatrixBlocks("lllsLL")
    def test2dMatrixBlocks_lllsLB(self): self.check2dMatrixBlocks("lllsLB")
    def test2dMatrixBlocks_lllsLA(self): self.check2dMatrixBlocks("lllsLA")

    def test2dMatrixBlocks_lllsAl(self): self.check2dMatrixBlocks("lllsAl")
    def test2dMatrixBlocks_lllsAL(self): self.check2dMatrixBlocks("lllsAL")
    def test2dMatrixBlocks_lllsAB(self): self.check2dMatrixBlocks("lllsAB")
    def test2dMatrixBlocks_lllsAA(self): self.check2dMatrixBlocks("lllsAA")

    def test2dMatrixBlocks_lllmLl(self): self.check2dMatrixBlocks("lllmLl")
    def test2dMatrixBlocks_lllmLL(self): self.check2dMatrixBlocks("lllmLL")
    def test2dMatrixBlocks_lllmLB(self): self.check2dMatrixBlocks("lllmLB")
    def test2dMatrixBlocks_lllmLA(self): self.check2dMatrixBlocks("lllmLA")

    def test2dMatrixBlocks_lllmAl(self): self.check2dMatrixBlocks("lllmAl")
    def test2dMatrixBlocks_lllmAL(self): self.check2dMatrixBlocks("lllmAL")
    def test2dMatrixBlocks_lllmAB(self): self.check2dMatrixBlocks("lllmAB")
    def test2dMatrixBlocks_lllmAA(self): self.check2dMatrixBlocks("lllmAA")

    def test2dMatrixBlocks_lllaLl(self): self.check2dMatrixBlocks("lllaLl")
    def test2dMatrixBlocks_lllaLL(self): self.check2dMatrixBlocks("lllaLL")
    def test2dMatrixBlocks_lllaLB(self): self.check2dMatrixBlocks("lllaLB")
    def test2dMatrixBlocks_lllaLA(self): self.check2dMatrixBlocks("lllaLA")

    def test2dMatrixBlocks_lllaAl(self): self.check2dMatrixBlocks("lllaAl")
    def test2dMatrixBlocks_lllaAL(self): self.check2dMatrixBlocks("lllaAL")
    def test2dMatrixBlocks_lllaAB(self): self.check2dMatrixBlocks("lllaAB")
    def test2dMatrixBlocks_lllaAA(self): self.check2dMatrixBlocks("lllaAA")

    def test2dMatrixBlocks_llllLl(self): self.check2dMatrixBlocks("llllLl")
    def test2dMatrixBlocks_llllLL(self): self.check2dMatrixBlocks("llllLL")
    def test2dMatrixBlocks_llllLB(self): self.check2dMatrixBlocks("llllLB")
    def test2dMatrixBlocks_llllLA(self): self.check2dMatrixBlocks("llllLA")

    def test2dMatrixBlocks_llllAl(self): self.check2dMatrixBlocks("llllAl")
    def test2dMatrixBlocks_llllAL(self): self.check2dMatrixBlocks("llllAL")
    def test2dMatrixBlocks_llllAB(self): self.check2dMatrixBlocks("llllAB")
    def test2dMatrixBlocks_llllAA(self): self.check2dMatrixBlocks("llllAA")

    def test2dMatrixBlocks_lllLLl(self): self.check2dMatrixBlocks("lllLLl")
    def test2dMatrixBlocks_lllLLL(self): self.check2dMatrixBlocks("lllLLL")
    def test2dMatrixBlocks_lllLLB(self): self.check2dMatrixBlocks("lllLLB")
    def test2dMatrixBlocks_lllLLA(self): self.check2dMatrixBlocks("lllLLA")

    def test2dMatrixBlocks_lllLAl(self): self.check2dMatrixBlocks("lllLAl")
    def test2dMatrixBlocks_lllLAL(self): self.check2dMatrixBlocks("lllLAL")
    def test2dMatrixBlocks_lllLAB(self): self.check2dMatrixBlocks("lllLAB")
    def test2dMatrixBlocks_lllLAA(self): self.check2dMatrixBlocks("lllLAA")

    def test2dMatrixBlocks_lllALl(self): self.check2dMatrixBlocks("lllALl")
    def test2dMatrixBlocks_lllALL(self): self.check2dMatrixBlocks("lllALL")
    def test2dMatrixBlocks_lllALB(self): self.check2dMatrixBlocks("lllALB")
    def test2dMatrixBlocks_lllALA(self): self.check2dMatrixBlocks("lllALA")

    def test2dMatrixBlocks_lllAAl(self): self.check2dMatrixBlocks("lllAAl")
    def test2dMatrixBlocks_lllAAL(self): self.check2dMatrixBlocks("lllAAL")
    def test2dMatrixBlocks_lllAAB(self): self.check2dMatrixBlocks("lllAAB")
    def test2dMatrixBlocks_lllAAA(self): self.check2dMatrixBlocks("lllAAA")

    def test2dMatrixBlocks_latsLl(self): self.check2dMatrixBlocks("latsLl")
    def test2dMatrixBlocks_latsLL(self): self.check2dMatrixBlocks("latsLL")
    def test2dMatrixBlocks_latsLB(self): self.check2dMatrixBlocks("latsLB")
    def test2dMatrixBlocks_latsLA(self): self.check2dMatrixBlocks("latsLA")

    def test2dMatrixBlocks_latsAl(self): self.check2dMatrixBlocks("latsAl")
    def test2dMatrixBlocks_latsAL(self): self.check2dMatrixBlocks("latsAL")
    def test2dMatrixBlocks_latsAB(self): self.check2dMatrixBlocks("latsAB")
    def test2dMatrixBlocks_latsAA(self): self.check2dMatrixBlocks("latsAA")

    def test2dMatrixBlocks_latmLl(self): self.check2dMatrixBlocks("latmLl")
    def test2dMatrixBlocks_latmLL(self): self.check2dMatrixBlocks("latmLL")
    def test2dMatrixBlocks_latmLB(self): self.check2dMatrixBlocks("latmLB")
    def test2dMatrixBlocks_latmLA(self): self.check2dMatrixBlocks("latmLA")

    def test2dMatrixBlocks_latmAl(self): self.check2dMatrixBlocks("latmAl")
    def test2dMatrixBlocks_latmAL(self): self.check2dMatrixBlocks("latmAL")
    def test2dMatrixBlocks_latmAB(self): self.check2dMatrixBlocks("latmAB")
    def test2dMatrixBlocks_latmAA(self): self.check2dMatrixBlocks("latmAA")

    def test2dMatrixBlocks_lataLl(self): self.check2dMatrixBlocks("lataLl")
    def test2dMatrixBlocks_lataLL(self): self.check2dMatrixBlocks("lataLL")
    def test2dMatrixBlocks_lataLB(self): self.check2dMatrixBlocks("lataLB")
    def test2dMatrixBlocks_lataLA(self): self.check2dMatrixBlocks("lataLA")

    def test2dMatrixBlocks_lataAl(self): self.check2dMatrixBlocks("lataAl")
    def test2dMatrixBlocks_lataAL(self): self.check2dMatrixBlocks("lataAL")
    def test2dMatrixBlocks_lataAB(self): self.check2dMatrixBlocks("lataAB")
    def test2dMatrixBlocks_lataAA(self): self.check2dMatrixBlocks("lataAA")

    def test2dMatrixBlocks_latlLl(self): self.check2dMatrixBlocks("latlLl")
    def test2dMatrixBlocks_latlLL(self): self.check2dMatrixBlocks("latlLL")
    def test2dMatrixBlocks_latlLB(self): self.check2dMatrixBlocks("latlLB")
    def test2dMatrixBlocks_latlLA(self): self.check2dMatrixBlocks("latlLA")

    def test2dMatrixBlocks_latlAl(self): self.check2dMatrixBlocks("latlAl")
    def test2dMatrixBlocks_latlAL(self): self.check2dMatrixBlocks("latlAL")
    def test2dMatrixBlocks_latlAB(self): self.check2dMatrixBlocks("latlAB")
    def test2dMatrixBlocks_latlAA(self): self.check2dMatrixBlocks("latlAA")

    def test2dMatrixBlocks_latLLl(self): self.check2dMatrixBlocks("latLLl")
    def test2dMatrixBlocks_latLLL(self): self.check2dMatrixBlocks("latLLL")
    def test2dMatrixBlocks_latLLB(self): self.check2dMatrixBlocks("latLLB")
    def test2dMatrixBlocks_latLLA(self): self.check2dMatrixBlocks("latLLA")

    def test2dMatrixBlocks_latLAl(self): self.check2dMatrixBlocks("latLAl")
    def test2dMatrixBlocks_latLAL(self): self.check2dMatrixBlocks("latLAL")
    def test2dMatrixBlocks_latLAB(self): self.check2dMatrixBlocks("latLAB")
    def test2dMatrixBlocks_latLAA(self): self.check2dMatrixBlocks("latLAA")

    def test2dMatrixBlocks_latALl(self): self.check2dMatrixBlocks("latALl")
    def test2dMatrixBlocks_latALL(self): self.check2dMatrixBlocks("latALL")
    def test2dMatrixBlocks_latALB(self): self.check2dMatrixBlocks("latALB")
    def test2dMatrixBlocks_latALA(self): self.check2dMatrixBlocks("latALA")

    def test2dMatrixBlocks_latAAl(self): self.check2dMatrixBlocks("latAAl")
    def test2dMatrixBlocks_latAAL(self): self.check2dMatrixBlocks("latAAL")
    def test2dMatrixBlocks_latAAB(self): self.check2dMatrixBlocks("latAAB")
    def test2dMatrixBlocks_latAAA(self): self.check2dMatrixBlocks("latAAA")

    def test2dMatrixBlocks_lansLl(self): self.check2dMatrixBlocks("lansLl")
    def test2dMatrixBlocks_lansLL(self): self.check2dMatrixBlocks("lansLL")
    def test2dMatrixBlocks_lansLB(self): self.check2dMatrixBlocks("lansLB")
    def test2dMatrixBlocks_lansLA(self): self.check2dMatrixBlocks("lansLA")

    def test2dMatrixBlocks_lansAl(self): self.check2dMatrixBlocks("lansAl")
    def test2dMatrixBlocks_lansAL(self): self.check2dMatrixBlocks("lansAL")
    def test2dMatrixBlocks_lansAB(self): self.check2dMatrixBlocks("lansAB")
    def test2dMatrixBlocks_lansAA(self): self.check2dMatrixBlocks("lansAA")

    def test2dMatrixBlocks_lanmLl(self): self.check2dMatrixBlocks("lanmLl")
    def test2dMatrixBlocks_lanmLL(self): self.check2dMatrixBlocks("lanmLL")
    def test2dMatrixBlocks_lanmLB(self): self.check2dMatrixBlocks("lanmLB")
    def test2dMatrixBlocks_lanmLA(self): self.check2dMatrixBlocks("lanmLA")

    def test2dMatrixBlocks_lanmAl(self): self.check2dMatrixBlocks("lanmAl")
    def test2dMatrixBlocks_lanmAL(self): self.check2dMatrixBlocks("lanmAL")
    def test2dMatrixBlocks_lanmAB(self): self.check2dMatrixBlocks("lanmAB")
    def test2dMatrixBlocks_lanmAA(self): self.check2dMatrixBlocks("lanmAA")

    def test2dMatrixBlocks_lanaLl(self): self.check2dMatrixBlocks("lanaLl")
    def test2dMatrixBlocks_lanaLL(self): self.check2dMatrixBlocks("lanaLL")
    def test2dMatrixBlocks_lanaLB(self): self.check2dMatrixBlocks("lanaLB")
    def test2dMatrixBlocks_lanaLA(self): self.check2dMatrixBlocks("lanaLA")

    def test2dMatrixBlocks_lanaAl(self): self.check2dMatrixBlocks("lanaAl")
    def test2dMatrixBlocks_lanaAL(self): self.check2dMatrixBlocks("lanaAL")
    def test2dMatrixBlocks_lanaAB(self): self.check2dMatrixBlocks("lanaAB")
    def test2dMatrixBlocks_lanaAA(self): self.check2dMatrixBlocks("lanaAA")

    def test2dMatrixBlocks_lanlLl(self): self.check2dMatrixBlocks("lanlLl")
    def test2dMatrixBlocks_lanlLL(self): self.check2dMatrixBlocks("lanlLL")
    def test2dMatrixBlocks_lanlLB(self): self.check2dMatrixBlocks("lanlLB")
    def test2dMatrixBlocks_lanlLA(self): self.check2dMatrixBlocks("lanlLA")

    def test2dMatrixBlocks_lanlAl(self): self.check2dMatrixBlocks("lanlAl")
    def test2dMatrixBlocks_lanlAL(self): self.check2dMatrixBlocks("lanlAL")
    def test2dMatrixBlocks_lanlAB(self): self.check2dMatrixBlocks("lanlAB")
    def test2dMatrixBlocks_lanlAA(self): self.check2dMatrixBlocks("lanlAA")

    def test2dMatrixBlocks_lanLLl(self): self.check2dMatrixBlocks("lanLLl")
    def test2dMatrixBlocks_lanLLL(self): self.check2dMatrixBlocks("lanLLL")
    def test2dMatrixBlocks_lanLLB(self): self.check2dMatrixBlocks("lanLLB")
    def test2dMatrixBlocks_lanLLA(self): self.check2dMatrixBlocks("lanLLA")

    def test2dMatrixBlocks_lanLAl(self): self.check2dMatrixBlocks("lanLAl")
    def test2dMatrixBlocks_lanLAL(self): self.check2dMatrixBlocks("lanLAL")
    def test2dMatrixBlocks_lanLAB(self): self.check2dMatrixBlocks("lanLAB")
    def test2dMatrixBlocks_lanLAA(self): self.check2dMatrixBlocks("lanLAA")

    def test2dMatrixBlocks_lanALl(self): self.check2dMatrixBlocks("lanALl")
    def test2dMatrixBlocks_lanALL(self): self.check2dMatrixBlocks("lanALL")
    def test2dMatrixBlocks_lanALB(self): self.check2dMatrixBlocks("lanALB")
    def test2dMatrixBlocks_lanALA(self): self.check2dMatrixBlocks("lanALA")

    def test2dMatrixBlocks_lanAAl(self): self.check2dMatrixBlocks("lanAAl")
    def test2dMatrixBlocks_lanAAL(self): self.check2dMatrixBlocks("lanAAL")
    def test2dMatrixBlocks_lanAAB(self): self.check2dMatrixBlocks("lanAAB")
    def test2dMatrixBlocks_lanAAA(self): self.check2dMatrixBlocks("lanAAA")

    def test2dMatrixBlocks_laNsLl(self): self.check2dMatrixBlocks("laNsLl")
    def test2dMatrixBlocks_laNsLL(self): self.check2dMatrixBlocks("laNsLL")
    def test2dMatrixBlocks_laNsLB(self): self.check2dMatrixBlocks("laNsLB")
    def test2dMatrixBlocks_laNsLA(self): self.check2dMatrixBlocks("laNsLA")

    def test2dMatrixBlocks_laNsAl(self): self.check2dMatrixBlocks("laNsAl")
    def test2dMatrixBlocks_laNsAL(self): self.check2dMatrixBlocks("laNsAL")
    def test2dMatrixBlocks_laNsAB(self): self.check2dMatrixBlocks("laNsAB")
    def test2dMatrixBlocks_laNsAA(self): self.check2dMatrixBlocks("laNsAA")

    def test2dMatrixBlocks_laNmLl(self): self.check2dMatrixBlocks("laNmLl")
    def test2dMatrixBlocks_laNmLL(self): self.check2dMatrixBlocks("laNmLL")
    def test2dMatrixBlocks_laNmLB(self): self.check2dMatrixBlocks("laNmLB")
    def test2dMatrixBlocks_laNmLA(self): self.check2dMatrixBlocks("laNmLA")

    def test2dMatrixBlocks_laNmAl(self): self.check2dMatrixBlocks("laNmAl")
    def test2dMatrixBlocks_laNmAL(self): self.check2dMatrixBlocks("laNmAL")
    def test2dMatrixBlocks_laNmAB(self): self.check2dMatrixBlocks("laNmAB")
    def test2dMatrixBlocks_laNmAA(self): self.check2dMatrixBlocks("laNmAA")

    def test2dMatrixBlocks_laNaLl(self): self.check2dMatrixBlocks("laNaLl")
    def test2dMatrixBlocks_laNaLL(self): self.check2dMatrixBlocks("laNaLL")
    def test2dMatrixBlocks_laNaLB(self): self.check2dMatrixBlocks("laNaLB")
    def test2dMatrixBlocks_laNaLA(self): self.check2dMatrixBlocks("laNaLA")

    def test2dMatrixBlocks_laNaAl(self): self.check2dMatrixBlocks("laNaAl")
    def test2dMatrixBlocks_laNaAL(self): self.check2dMatrixBlocks("laNaAL")
    def test2dMatrixBlocks_laNaAB(self): self.check2dMatrixBlocks("laNaAB")
    def test2dMatrixBlocks_laNaAA(self): self.check2dMatrixBlocks("laNaAA")

    def test2dMatrixBlocks_laNlLl(self): self.check2dMatrixBlocks("laNlLl")
    def test2dMatrixBlocks_laNlLL(self): self.check2dMatrixBlocks("laNlLL")
    def test2dMatrixBlocks_laNlLB(self): self.check2dMatrixBlocks("laNlLB")
    def test2dMatrixBlocks_laNlLA(self): self.check2dMatrixBlocks("laNlLA")

    def test2dMatrixBlocks_laNlAl(self): self.check2dMatrixBlocks("laNlAl")
    def test2dMatrixBlocks_laNlAL(self): self.check2dMatrixBlocks("laNlAL")
    def test2dMatrixBlocks_laNlAB(self): self.check2dMatrixBlocks("laNlAB")
    def test2dMatrixBlocks_laNlAA(self): self.check2dMatrixBlocks("laNlAA")

    def test2dMatrixBlocks_laNLLl(self): self.check2dMatrixBlocks("laNLLl")
    def test2dMatrixBlocks_laNLLL(self): self.check2dMatrixBlocks("laNLLL")
    def test2dMatrixBlocks_laNLLB(self): self.check2dMatrixBlocks("laNLLB")
    def test2dMatrixBlocks_laNLLA(self): self.check2dMatrixBlocks("laNLLA")

    def test2dMatrixBlocks_laNLAl(self): self.check2dMatrixBlocks("laNLAl")
    def test2dMatrixBlocks_laNLAL(self): self.check2dMatrixBlocks("laNLAL")
    def test2dMatrixBlocks_laNLAB(self): self.check2dMatrixBlocks("laNLAB")
    def test2dMatrixBlocks_laNLAA(self): self.check2dMatrixBlocks("laNLAA")

    def test2dMatrixBlocks_laNALl(self): self.check2dMatrixBlocks("laNALl")
    def test2dMatrixBlocks_laNALL(self): self.check2dMatrixBlocks("laNALL")
    def test2dMatrixBlocks_laNALB(self): self.check2dMatrixBlocks("laNALB")
    def test2dMatrixBlocks_laNALA(self): self.check2dMatrixBlocks("laNALA")

    def test2dMatrixBlocks_laNAAl(self): self.check2dMatrixBlocks("laNAAl")
    def test2dMatrixBlocks_laNAAL(self): self.check2dMatrixBlocks("laNAAL")
    def test2dMatrixBlocks_laNAAB(self): self.check2dMatrixBlocks("laNAAB")
    def test2dMatrixBlocks_laNAAA(self): self.check2dMatrixBlocks("laNAAA")

    def test2dMatrixBlocks_lalsLl(self): self.check2dMatrixBlocks("lalsLl")
    def test2dMatrixBlocks_lalsLL(self): self.check2dMatrixBlocks("lalsLL")
    def test2dMatrixBlocks_lalsLB(self): self.check2dMatrixBlocks("lalsLB")
    def test2dMatrixBlocks_lalsLA(self): self.check2dMatrixBlocks("lalsLA")

    def test2dMatrixBlocks_lalsAl(self): self.check2dMatrixBlocks("lalsAl")
    def test2dMatrixBlocks_lalsAL(self): self.check2dMatrixBlocks("lalsAL")
    def test2dMatrixBlocks_lalsAB(self): self.check2dMatrixBlocks("lalsAB")
    def test2dMatrixBlocks_lalsAA(self): self.check2dMatrixBlocks("lalsAA")

    def test2dMatrixBlocks_lalmLl(self): self.check2dMatrixBlocks("lalmLl")
    def test2dMatrixBlocks_lalmLL(self): self.check2dMatrixBlocks("lalmLL")
    def test2dMatrixBlocks_lalmLB(self): self.check2dMatrixBlocks("lalmLB")
    def test2dMatrixBlocks_lalmLA(self): self.check2dMatrixBlocks("lalmLA")

    def test2dMatrixBlocks_lalmAl(self): self.check2dMatrixBlocks("lalmAl")
    def test2dMatrixBlocks_lalmAL(self): self.check2dMatrixBlocks("lalmAL")
    def test2dMatrixBlocks_lalmAB(self): self.check2dMatrixBlocks("lalmAB")
    def test2dMatrixBlocks_lalmAA(self): self.check2dMatrixBlocks("lalmAA")

    def test2dMatrixBlocks_lalaLl(self): self.check2dMatrixBlocks("lalaLl")
    def test2dMatrixBlocks_lalaLL(self): self.check2dMatrixBlocks("lalaLL")
    def test2dMatrixBlocks_lalaLB(self): self.check2dMatrixBlocks("lalaLB")
    def test2dMatrixBlocks_lalaLA(self): self.check2dMatrixBlocks("lalaLA")

    def test2dMatrixBlocks_lalaAl(self): self.check2dMatrixBlocks("lalaAl")
    def test2dMatrixBlocks_lalaAL(self): self.check2dMatrixBlocks("lalaAL")
    def test2dMatrixBlocks_lalaAB(self): self.check2dMatrixBlocks("lalaAB")
    def test2dMatrixBlocks_lalaAA(self): self.check2dMatrixBlocks("lalaAA")

    def test2dMatrixBlocks_lallLl(self): self.check2dMatrixBlocks("lallLl")
    def test2dMatrixBlocks_lallLL(self): self.check2dMatrixBlocks("lallLL")
    def test2dMatrixBlocks_lallLB(self): self.check2dMatrixBlocks("lallLB")
    def test2dMatrixBlocks_lallLA(self): self.check2dMatrixBlocks("lallLA")

    def test2dMatrixBlocks_lallAl(self): self.check2dMatrixBlocks("lallAl")
    def test2dMatrixBlocks_lallAL(self): self.check2dMatrixBlocks("lallAL")
    def test2dMatrixBlocks_lallAB(self): self.check2dMatrixBlocks("lallAB")
    def test2dMatrixBlocks_lallAA(self): self.check2dMatrixBlocks("lallAA")

    def test2dMatrixBlocks_lalLLl(self): self.check2dMatrixBlocks("lalLLl")
    def test2dMatrixBlocks_lalLLL(self): self.check2dMatrixBlocks("lalLLL")
    def test2dMatrixBlocks_lalLLB(self): self.check2dMatrixBlocks("lalLLB")
    def test2dMatrixBlocks_lalLLA(self): self.check2dMatrixBlocks("lalLLA")

    def test2dMatrixBlocks_lalLAl(self): self.check2dMatrixBlocks("lalLAl")
    def test2dMatrixBlocks_lalLAL(self): self.check2dMatrixBlocks("lalLAL")
    def test2dMatrixBlocks_lalLAB(self): self.check2dMatrixBlocks("lalLAB")
    def test2dMatrixBlocks_lalLAA(self): self.check2dMatrixBlocks("lalLAA")

    def test2dMatrixBlocks_lalALl(self): self.check2dMatrixBlocks("lalALl")
    def test2dMatrixBlocks_lalALL(self): self.check2dMatrixBlocks("lalALL")
    def test2dMatrixBlocks_lalALB(self): self.check2dMatrixBlocks("lalALB")
    def test2dMatrixBlocks_lalALA(self): self.check2dMatrixBlocks("lalALA")

    def test2dMatrixBlocks_lalAAl(self): self.check2dMatrixBlocks("lalAAl")
    def test2dMatrixBlocks_lalAAL(self): self.check2dMatrixBlocks("lalAAL")
    def test2dMatrixBlocks_lalAAB(self): self.check2dMatrixBlocks("lalAAB")
    def test2dMatrixBlocks_lalAAA(self): self.check2dMatrixBlocks("lalAAA")


if __name__ == '__main__':
    unittest.main()

