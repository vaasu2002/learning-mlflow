

Run server on another termial
```bash
mlflow server  \
--backend-store-uri sqlite:///mlflow.db  \
--default-artifact-root ./artifacts  \
--host 127.0.0.1 -p 1234
```

Running the code with default parameters (on another termial)
```bash
python demo/demo.py
```

Running the code with new parameters
```bash
python demo/demo.py -0.7 0.001
```

Opening the
```bash
mlflow ui
```


Serving on http://127.0.0.1:5000