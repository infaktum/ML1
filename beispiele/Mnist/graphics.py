import numpy as np 
import matplotlib.pyplot as plt

def show_network_result(row):
    data = test_data_list[row].split(',')[1:]
    value =  test_data_list[row][0]
    output = network.forward((np.asfarray(data) / 255.0 * 0.99) + 0.01)
    show_output(output,value)
    pass

def show_output(output,value):
    y_pos = np.arange(len(output))
    
    plt.bar(y_pos, output, align='center', alpha=0.5)
    plt.xticks(range(9))
    plt.xlabel('Output-Neuronen')
    plt.ylabel('Wert des Outputs')    
    plt.title(value )
    
    plt.show()
    pass

def show_number(data):
    """
    Plottet den Eintrag mit dem angegebenen Index.
    """
    for r in range(data):
        for c in range(cols):
            plt.subplot(rows,cols,r*rows + c + 1)
            index = r*rows + c + offset
            label = self.data[index,0]
            bild_daten = self.data[index,1:].reshape(28,28)
            #plt.title(f'Number = {label}')                
            plt.imshow(bild_daten,cmap='Blues')