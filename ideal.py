
"""
ideal
=====

"""

#############################################################################
#  Copyright (C) 2015, 2016                                                 #
#                Manuel Kauers (mkauers@gmail.com),                         #
#                                                                           #
#  Distributed under the terms of the GNU General Public License (GPL)      #
#  either version 2, or (at your option) any later version                  #
#                                                                           #
#  http://www.gnu.org/licenses/                                             #
#############################################################################

import nullspace 

from datetime import datetime
from sage.rings.noncommutative_ideals import *

class OreLeftIdeal(Ideal_nc):

    def __init__(self, ring, gens, coerce=True):
        Ideal_nc.__init__(self, ring, gens, coerce, "left")

    def gb(self, infolevel=0):
        """
        Returns the Groebner basis of this ideal with respect to the 
        monomial order associated to the parent.
        """
        try:
            return self.__gb
        except:
            pass

        ## TODO: switch to polynomial ring and clear denominators of generators
        ## TODO: implement sugar strategy?
        
        # ~~~ relatively naive code ~~~

        # tools
        def info(i, msg):
            if infolevel >= i:
                print msg

        X = self.ring().gens() 
        def maketerm(e): # exponent vector to monomial
            return prod( x**i for x, i in zip(X, e) )

        def lcm(u, v): # exponent vector of lcm(lt(u),lt(v))
            return u.exp().emax(v.exp())
        
        def makepair(u, v): # first entry is lcm(lt(u),lt(v))
            ht = maketerm(lcm(u, v))
            s = max(ht.tdeg() - u.lm().tdeg() + u.sugar, ht.tdeg() - v.lm().tdeg() + v.sugar)
            return (ht, u, v, s)

        def update(G, B, h): # inspired by alg UPDATE in BWK p 230, but without using the first criterion 
            if h.is_zero():
                info(2, "new polynomial is zero")
                return G, B
            if False:
                return G + [h], B + [makepair(g, h) for g in G]                    
            info(2, "new polynomial has lm=" + str(h.lm()) + ", determine new pairs...")
            C = [g for g in G]; C.sort(cmp=lambda u,v: -1 if (u.lm() + v.lm()).lm() == u.lm() else 1) # smallest leading term last
            # 1. discard the pairs (C[i],h) for which there is another pair (C[j], h) with lt(C[j],h)|lt(C[i],h)
            for i in range(len(C)): 
                for j in range(len(C)):
                    if C[j] is not None and i != j and min(lcm(C[i],h).esub(lcm(C[j],h))) >= 0:
                        C[i] = None
                        break
            C = [makepair(g, h) for g in C if g is not None]
            # 2. discard an old pair (u,v) if (u,h,v) is a bb-pair but (h,u,v) and (u,v,h) are not
            for b in B:
                if min(b[0].exp().esub(h.exp())) < 0 or b[0].exp() in [lcm(h, b[1]), lcm(h, b[2])]:
                    C.append(b)
            C.sort(key=lambda c: -c[-1]) # smallest pair last
            # 3. update basis
            G = [g for g in G if min(g.exp().esub(h.exp())) < 0] + [h]
            return G, C

        # initialization
        info(1, "initialization...")
        G = []; # current basis, sorted such that smallest leading term comes first
        C = []; # current list of critical pairs, sorted such that smallest lcm comes last
        for g in self.gens():
            g.sugar = g.tdeg()
            G, C = update(G, C, g.reduce(G, normalize=True, coerce=False))

        # buchberger loop
        info(1, "main loop...")
        while len(C) > 0:
            t, u, v, s = C.pop()
            info(2, datetime.today().ctime() + ": " + str(len(C) + 1) + " pairs left; taking pair with lcm(lm,lm)=" + str(t))            
            uterm = v.lc()*maketerm(t.exp().esub(u.exp()))
            vterm = u.lc()*maketerm(t.exp().esub(v.exp()))
            spol = uterm*u - vterm*v
            spol.sugar = s
            G, C = update(G, C, spol.reduce(G, normalize=True, infolevel=infolevel-2, coerce=False))

        # autoreduction
        info(2, "autoreduction...")
        for i in range(len(G)): 
            G[i] = G[i].reduce(G[:i] + G[i+1:], normalize=True, infolevel=infolevel-3, coerce=False)
        G = [g for g in G if not g.is_zero()]
            
        # done.
        info(1, "completion completed, Groebner basis has " + str(len(G)) + " elements.")
        self.__gb = G
        return G
        
