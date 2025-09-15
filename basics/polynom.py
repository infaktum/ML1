import numpy as np

class Polynom:
    """ Einige Funktionen zur Erzeugung und Darstellung von Polynomen """

    def __init__(self,coeff:np.ndarray):
        """ Erzeugt ein Polynom mit den Koeffizienten aus a """
        self.coeff = coeff
        self.p = lambda x: np.sum([c * x**k for k,c in enumerate(coeff)])
        
    def evaluate(self,x:np.ndarray):
        return self.p(x)

    def to_text(self):
        """ Ausgabe des Polynoms in Text-Form. """
        comp = []
        text = "P(x) = "
        for n,a in enumerate(self.coeff):   
            c = "{:.3f}".format(a)
            if n > 0:
                c += "x"
            if n > 1:
                c += "^" + str(n)
            comp.append(c)

        comp.reverse()
        for c in comp:
            text += c + " + "
        text = text[:-3]
        return text
    
    def to_latex(self):
        """ Ausgabe des Polynoms in LaTeX-Form. """
        return "$" + self.to_text() + "$"    

    def plot(self,plot,x_min,x_max):
        """ Grafische Ausgabe eines Polynoms """ 
        subplot = plot.subplot(1,1,1)
        subplot.set_xlabel('$x$')
        subplot.set_ylabel('$P(x)$')
        subplot.set_title(self.to_latex())
        subplot.grid(True)
        xp = np.linspace(x_min,x_max,100)
        yp = [self.p(x) for x in xp]
        #plot.plot([0,0],[np.min(yp),np.max(yp)],c='k')
        subplot.plot(xp,yp)
        pass

if __name__ == "__main__":
    print("Polynom-Modul für KI-Kurs")