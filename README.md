# Glass Cut Predictor
Vorhersage Tool für Glaszuschnitte bei Floatglas-Slumping



Problem 
Bei Glasslumping vorherzusagen wie sich ein Glaszuschnitt auf die gewünschte Negativform legt ist extrem Erfahrungsabhängig und Kompliziert vorherzusagen. Besonders Kompliziert ist es da viele Parameter zusammenspielen darunter Glasdicke, Negativform, Temperaturprogramm und reales Materialverhalten. 


Projektidee
Einheitliche Dokumentation der Negativform, des Formzuschnittes, des Materials und der Brennkurve vor dem Brandt und dazu die entstandene Glasgeometrie. Für eine neue Form wird  dann in den Daten abgeglichen wie der Zuschnitt, die Brennkurve und evlt das Material ähnlicher Formen funktioniert hat. (Case-Based Reasoning)


Vorhersageansatz:
- radialer Zuschnitt aus Negativform bestimmen (geometrische berechnung)
- Negativform mit Formen im Datensatz abgleichen (mithilfe von Merkmalsvektor)
- Zuschnitt ähnlicher Formen + radialer Zuschnitt = finale Zuschnitt

Datensatz:

1 Datenpunkt
- 3d Form / heightmap
- verwendeter Zuschnitt 
- Brandergebnis




## Install and run (Windows / Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
