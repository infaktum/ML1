# Maschinelles Lernen 1

## Einführung in die Grundlagen des Maschinenlernens mit Python und Jupyter-Notebooks

Die Notebooks in diesem Paket stammen aus meinen Kursen zur Künstlichen Intelligenz.


### Ziel des Kurses

Der Kurs soll die mathematischen Grundlagen zu den verschiedenen Verfahren der Künstlichen Intelligenz liefern und diese mit Hilfe frei zugänglicher Werkzeuge vorstellen.


### Aufbau und Inhalt

Mathematische Erläuterungen und Live-Coding wechseln einander ab. Es existieren keine Folien; 
die Notebooks enthalten den gesamten Code und die dazugehörenden Erklärungen.

Die zum Einsatz kommende Mathematik ist nicht sonderlich komplex; sie kann trotzdem nicht vollständig in die Tiefe 
gehend behandelt werden. Im Internet, insbesondere bei der Wikipedia, sind besser Darstellungen zu finden.
Zu den behandelten Themen gehören:

* Lineare Regression
* Klassifizierungsverfahren wie _k-means_
* Entscheidungsbäume
* Klassifizierung des Iris-Datensatzes durch unterschiedliche Verfahren
* Das Perzeptron 
* Das Gradientenabstiegsverfahren
* Zeichenerkennung beim MNIST-Datensatz durch ein KNN

##### 1. Die programmiertechnischen Werkzeuge 

Als Programmiersprache hat sich __Python__ in der Künstlichen Intelligenz einen festen Platz gesichert. 
Dies hat verschiedene Gründe. Bei der Entwicklung von Python hatte Einfachheit und Pragmatismus immer 
Vorrang vor einem festen Programmier-Paradigma. So bietet etwa auch die Sprache __Java__ mittlerweile
_funktionale Programmierung_ und _Lambdas_ sowie eine Palette verschiedener Datenstrukturen,
doch steht hier immer die Objekt-Orientierung im Vordergrund und manchmal eben auch im Weg.
Python ist hier völlig undogmatisch; Algorithmen und Datenstrukturen sind viel klarer herausgearbeitet,
und Funktionen sind wirklich _first class citizens_.

Darüber hinaus bietet die Technologie der __Jupyter-Notebooks__ nicht nur die Möglichkeit,
interaktiv schnelle Lösung interaktiv  zu erarbeiten. Die Notebooks selbst können hinterher
auch zur Dokumentation und Demonstration verwendet werden. Die vorhandenen Python-Bibliotheken,
allen voran _matplotlib_, ermöglichen es, die unbedingt notwendigen grafischen Darstellungen von
Daten und Ergebnissen mit nur weniger Zeilen Code zu erstellen, als etwa zu einem einfachen
"Hello, world" in einer OOP-Sprache wie Java notwendig sind.

`Scikit-learn` ist eine weit verbreitete Open-Source-Bibliothek für maschinelles Lernen in Python.
Sie bietet eine Vielzahl von Algorithmen für überwachtes und unüberwachtes Lernen und basiert auf
anderen wichtigen Bibliotheken wie `NumPy`, `SciPy` und `Matplotlib`. Die Modelle verfügen über einheitliche
Schnittstellen mit Methoden wie `fit()`, `predict()` und `score()`. Durch diese Vereinheitlichung erleichtert
sie den Einstieg in die KI und das Experimentieren mit verschiedenen Modellen und führt schnell zu erstaunlichen Ergebnisse.


##### 2. Grundbegriffe der Künstlichen Intelligenz 

Während der Arbeit mit den einzelnen Verfahren werden auch die verschiedenen Begriffe der KI vorgestellt wie

* Modell
* Bias
* Supervised und unsupervised Learning
* Regression und Klassifikation
* Kostenfunktion
* Maschinelles lernen


##### 3. Die mathematischen Grundlagen

Zum Verständnis der Verfahreb sind lediglich einfache Kenntnisse der (multidimensionalen) Analysis und Linearen Algebra notwendig, die in diesem Kurs ebenfalls vermittelt werden und zum größten Teil Thema eines Mathematik-Grundkurses der Oberstufe sind. Dies sind u.a.:

* Vektoren und Matrizen
* Differentiation
* Gradientenabstieg


### Der Quellcode

Der Python-Code in diesem Kurs wurde bewusst einfach gehalten; Lesbarkeit und verständlichkeit 
stehen im Vordergrund.Neben `scikit-learn` werden vor allem die beiden Python-Bibliotheken `NumPy`
(schnellere numerische Berechnungen) und `matplotlib` (Visualisierung) vorgestellt. 

Der Python-Code ist in keiner Weise optimiert, um höhere Geschwindigkeit zu erzielen. 
Trotzdem ist es mit ihm möglich, die bekannte Aufgabe der Ziffernerkennung des MNIST-Zeichensatzes auf einem 
einfachen Laptop ohne Spezialprozessor in wenigen Sekunden Rechenzeit zu lösen. 
Darauf aufbauend können Teilnehmer des Kurses bei Interesse(!) einfach eigene Experimente vornehmen, 
bevor sie dann vielleicht auf die schlagkräftigeren Werkzeuge umsteigen.


