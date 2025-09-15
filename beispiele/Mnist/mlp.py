import numpy as np

###############################################################################################

class Perceptron:
    """ Ein generisches Perzeptron """
    
    def __init__(self,*nodes: list[int]) -> None:
        """ Initialisert das Perzeptron. Gewichte und Bias werden auf zufällige Werte zwischen -1 und +1 gesetzt. """
        input, output = nodes
        self.w = np.random.uniform(-1,1,(input,output)).T
        self.b = np.random.uniform(-1,1,output)    
        pass

    def train(self,input: list[int],target: list[int],alpha:float = 0.1,epochs: int = 1) ->  None:
        """ 
        Trainiert das Perzeptron mit den übergebenen Daten input und den Zielwerten target gemäß dem Lernalgorithmus. 
        alpha bestimmt die Lernrate und epochs die Anzahl der Durchläufe. 
        """
        for _ in range(epochs):
            for x, t in zip(input,target):  
                error = (t - self.forward(x))     
                self.w += alpha * np.outer(error,x)
                self.b += alpha * error
        pass

    def forward(self,x: list[int]) -> list[int]:
        """ 
        Berechnet das Ergebnis bei einem angelegten Input x. 
        Durch die Aktivierungsfunktion sind nur die Werte 0 (inaktiv) und 1 (aktiv) möglich.
        """
        return self.activation(np.dot(self.w,x) + self.b) 

    def activation(self,signal: list[int]) -> int:
        """ Die Aktivierungsfunktion testet, obL das Signal positiv ist (Perzeptron-Regel). """
        return (signal > 0).astype(int)    
   
    def test(self,input: list[int],target: list[int]) -> list[bool]:
        """ 
        Testet die Funktion des Perzeptrons, indem alle durch die Eingabewerte input erzeugten Ausgaben mit den 
        Zielwerten target verglichen werden
        """
        y = self.forward(input)
           
    def __str__(self) -> str:
        return "Gewichte: {0}, Bias = {1}".format(self.w, self.b)

#####################################################################################################
 
    
class MLP:
    def __init__(self, *nodes: list[int]) -> None:
        ''' Setzen der Parameter des MLP. Gewichte werden zufällig erzeugt. '''
        self.inodes, self.hnodes, self.onodes = nodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.transfer = lambda x: 1/(1 + np.exp(-x)) # Die Sigmoid-Funktion
        pass

    def fit(self, inputs_list: np.ndarray, targets_list: np.ndarray,lr: float = 0.2) -> None :
        ''' Training des Neuronalen Netzwerks '''
   
        inputs =  np.transpose(np.array(inputs_list, ndmin=2))
        targets = np.transpose(np.array(targets_list, ndmin=2))

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.transfer(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.transfer(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # Backpropagation
        self.who += lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
    
        pass

    def forward(self, inputs_list: np.ndarray):
        ''' Abfrage des Neuronalen Netzwerks '''
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.transfer(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.transfer(final_inputs)
        return np.concatenate(final_outputs).ravel()

    def predict(self, inputs: np.ndarray) -> int:
        return np.argmax(self.forward(inputs))

    def performance(self,test_data):
        '''
        Testet die Leistung des Neuronalen Netzwerkes mit Hilfe von Testdaten. 
        Es wird ein Wert fuer die Zuverlaessigkeit und die Liste der falschen Zuordnungen zurueckgegeben.
        '''
        fails = []
        for n,record in enumerate(test_data):
            correct_label = int(record[0])
            inputs = (np.asfarray(record[1:]) / 255.0 * 0.98) + 0.01
            outputs = self.forward(inputs)
            label = np.argmax(outputs)
            if (label != correct_label):
                fails.append(n)
        performance =  1. - (len(fails) / len(test_data))
        return performance , fails

    def save(self,file: str) -> None:
        '''Speichert die Gewichte des Netzwerks'''
        with open(file + '.npy', 'wb') as f:
            np.save(f,self.wih, allow_pickle=True)
            np.save(f,self.who, allow_pickle=True)
        print("Gewichte wurden gespeichert")            

    def load(self,file: str) -> None:
        '''Lädt die Gewichte des Netzwerks'''        
        with open(file + '.npy', 'rb') as f:
            self.wih = np.load(f)
            self.who = np.load(f)
        print("Gewichte wurden geladen")      
        
    def __str__(self) -> str:
        return "in -> hidden:" + np.array2string(self.wih) +"\nhidden -> out" + np.array2string(self.who) 


#########################################################################################

logit = lambda x : np.log( x / (1-x))

class TwoWayMLP(MLP):
    ''' Erweitert das Neuronale Netzwerk um eine Reverse-Funktion '''
    def reverse(self, target: np.ndarray) -> np.ndarray:
        output = np.array(target, ndmin=2).T

        final_inputs = logit(output)

        hidden_outputs = np.dot(self.who.T, final_inputs)

        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        hidden_inputs = logit(hidden_outputs)

        inputs = np.dot(self.wih.T, hidden_inputs)
        # Skalierung auf Grauwerte
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs


###################

def performance(network,test_data):
    '''
    Testet die Leistung des Neuronalen Netzwerkes mit Hilfe von Testdaten. 
    Es wird ein Wert fuer die Zuverlaessigkeit und die Liste der falschen Zuordnungen zurueckgegeben.
    '''
    fails = []
    for n,row in enumerate(test_data):
        data = row.split(',')
        correct_label = int(data[0])

        inputs = (np.asfarray(data[1:]) / 255.0 * 0.99) + 0.01
        outputs = network.forward(inputs)

        label = np.argmax(outputs)

        if (label != correct_label):
            fails.append(n)
    pass
    performance = 1- len(fails) / len(test_data)
 
    return performance , fails


####################################

if __name__ == "__main__":
    print("Generisches Neuronales Netz.")