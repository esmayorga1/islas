import pandas as pd
import os


class AirTemperatureMonthly:
    """
    Lee datos diarios de temperatura del aire desde un CSV
    y calcula promedios mensuales por año.
    
    Se usa como insumo temporal (no espacial) para el modelo de datos sintéticos.
    """

    def __init__(self, csv_path, sep=";"):
        self.csv_path = csv_path
        self.sep = sep
        self.df = None
        self.monthly = None

    # -----------------------------------------------------------
    # 1. Cargar datos
    # -----------------------------------------------------------
    def load(self):
        """Carga y estandariza el CSV."""
        df = pd.read_csv(self.csv_path, sep=self.sep)

        required_cols = ["Fecha inicial", "TAire2"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columnas faltantes en CSV: {missing}")

        df["Fecha inicial"] = pd.to_datetime(df["Fecha inicial"], errors="raise")
        df = df.rename(columns={"TAire2": "T_air"})
        df = df.dropna(subset=["T_air"])

        self.df = df
        return df

    # -----------------------------------------------------------
    # 2. Promedio mensual
    # -----------------------------------------------------------
    def compute_monthly(self):
        """Crea tabla con promedio mensual por año."""
        if self.df is None:
            self.load()

        df = self.df.copy()
        df["year"] = df["Fecha inicial"].dt.year.astype(int)
        df["month"] = df["Fecha inicial"].dt.month.astype(int)

        monthly = (
            df.groupby(["year", "month"])["T_air"]
            .mean()
            .reset_index()
            .sort_values(["year", "month"])
        )

        self.monthly = monthly
        return monthly

    # -----------------------------------------------------------
    # 3. Convertir a diccionario
    # -----------------------------------------------------------
    def to_dict(self, key_format="yyyymm"):
        """
        Devuelve diccionario mensual:

        "tuple"   -> {(2021, 1): 15.2}
        "yyyymm"  -> {"2021-01": 15.2}
        """
        if self.monthly is None:
            self.compute_monthly()

        result = {}

        for _, row in self.monthly.iterrows():
            y = int(row["year"])
            m = int(row["month"])
            t = float(round(row["T_air"], 4))

            if key_format == "tuple":
                key = (y, m)

            elif key_format == "yyyymm":
                key = f"{y}-{m:02d}"

            else:
                raise ValueError("Formato de llave no reconocido")

            result[key] = t

        return result

    # -----------------------------------------------------------
    # 4. Guardar CSV mensual
    # -----------------------------------------------------------
    def save_csv(self, output_path):
        """Guarda el archivo mensual en CSV."""
        if self.monthly is None:
            self.compute_monthly()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.monthly.to_csv(output_path, index=False, sep=";")

        return output_path
