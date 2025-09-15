import numpy as np
import matplotlib.pyplot as plt

class MnistDataset:    
    """ Wrapper-Klasse für den MNIST-Datensatz. """
    def __init__(self,csv_file,rows=28,cols=28):  
        with open(csv_file, "r") as file: 
            self.data = [line.split(',') for line in file.readlines()]
        pass
        self.rows,self.cols = rows, cols
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        """ Liefert den Eintrag mit dem angegebenen Index. """        
        return self.data[index]
    
    def values(self):     
        """ Liefert die Rohdaten, also N Zeilen mit 785 Werten """
        return self.data

    
    def X(self):     
        """ Liefert die Bild-Rohdaten """
        return self.data[1:]

    
    def y(self):     
        """ Liefert den Zahlenwert """
        return self.data[0]
    
    def plot(self,index):
        """
        Plottet den Eintrag mit dem angegebenen Index.
        """
        row = self.data[index]
        label = row[0]
        img = np.reshape(row[1:],(28,28)).astype(float)

        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.title(f'Number = {label}')
        plt.imshow(img,cmap='Blues')  
        plt.show()
        pass

    def plot_list(self,rows,cols,offset = 0):
        """
        Plottet den Eintrag mit dem angegebenen Index.
        """
        for r in range(rows):
            for c in range(cols):
                plt.subplot(rows,cols,r*cols + c +1)
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                row = self.data[r*rows + c + offset]
                label = row[0]
                img = np.reshape(row[1:],(self.rows,self.cols)).astype(float)
                #plt.title(f'Number = {label}')                
                plt.imshow(img,cmap='Blues')
        pass

    @classmethod   
    def get_scaler(cls):
        return lambda record: (np.asfarray(record[1:]) / 255.0 * 0.98) + 0.01

    @classmethod     
    def trainings_daten(cls):
        return cls('daten/mnist_train.csv')
    @classmethod 
    def test_daten(cls):
        return cls('daten/mnist_test.csv')  


if __name__ == "__main__":
    print("MNist-Modul für KI-Kurs")