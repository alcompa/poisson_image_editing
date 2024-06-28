

# Poisson Image Editing

Implementation of seamless cloning methods ("Importing Gradients", "Mixing Gradients") from [Poisson Image Editing, Pérez et al](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Perez03.pdf).

For a detailed explanation, see the [project report](report/report_en.pdf).

## Results
![bear](report/insertion_bear.png)
![ginevra](report/ginevra_insertion.png)
![equation](report/holes_equation.png)
![rainbow](report/rainbow_transparent.png)
![canadair](report/canadair_insertion.png)

## Virtual Env Setup
```
python -m venv .poissonenv
.poissonenv\Scripts\activate
python -m pip install -r requirements.txt
```