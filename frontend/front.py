import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from metodos_front import ajuste_lineal 


st.set_page_config(layout="wide")
st.title("Gráfica desde Excel")


modo = st.radio(
    "Entrada de datos",
    ["Subir Excel", "Pegar datos"]
)

df = None

@st.cache_data
def load_sheet(file, sheet):
    excel = pd.ExcelFile(file)
    return pd.read_excel(excel, sheet_name=sheet)


if modo == "Subir Excel":
    file = st.file_uploader("Sube un Excel", type="xlsx")

    if file:
        excel = pd.ExcelFile(file)
        n_hojas = len(excel.sheet_names)

        indice = st.number_input(
            "Número de hoja",
            min_value=1,
            max_value=n_hojas,
            step=1
        )

        df = load_sheet(file, indice-1)

# --- Texto ---
if modo == "Pegar datos":
    texto = st.text_area("Pega los datos copiados desde Excel")

    if texto.strip():
        df = pd.read_csv(io.StringIO(texto), sep="\t")



# --- Si hay datos ---
if df is not None:

    # Formateamos en caso de no tener titulo en la columna
    k = 1
    nuevasc = []
    for i, columna in enumerate(df.columns):

        if 'Unnamed' in str(columna) :
            nuevasc.append(f"No title {k}")
            k += 1
        else:
            nuevasc.append(columna)

    df.columns = nuevasc

    st.subheader("Vista del DataFrame")
    st.dataframe(df)

    st.divider()
    st.sidebar.subheader("Configuración de la gráfica")

    # Creamos el setup de columnas para la grafica
    col1, col3, col2 = st.columns([1, 1, 3])

    # Columna 1 de seleccion de columnas
    with col1:
        x = st.selectbox("Columna X", df.columns)
        y = st.selectbox("Columna Y", df.columns)
        

    valores = df[x].tolist()


    errx = None
    erry = None

    usar_errx = st.sidebar.checkbox("Error en X")
    usar_erry = st.sidebar.checkbox("Error en Y")

    if usar_errx:
        errx = st.sidebar.selectbox("Columna error X", df.columns)

    if usar_erry:
        erry = st.sidebar.selectbox("Columna error Y", df.columns)


    # Calculamos el ajuste lineal si procede
    with col1:
        ajuste = st.sidebar.checkbox("Ajuste lineal")
        m,b,sm,sb, cfit = ajuste_lineal(df[x], df[y], df[erry])


    st.sidebar.subheader("Opciones de formato")

    usar_titulo = st.sidebar.checkbox("Añadir título")
    usar_ejes = st.sidebar.checkbox("Personalizar ejes")
    usar_leyenda = st.sidebar.checkbox("Añadir leyenda")
    usar_grid = st.sidebar.checkbox("Añadir grid")    

    if usar_titulo:
        titulo = st.text_input("Título (puede usar LaTeX)")

    if usar_ejes:
        xlabel = st.text_input("Etiqueta eje X")
        ylabel = st.text_input("Etiqueta eje Y")

    if usar_leyenda:
        legend = st.text_input("Texto de leyenda")

        
    fig, ax = plt.subplots(figsize = (8,5), dpi = 500)

    if errx or erry:
        ax.errorbar(
            df[x],
            df[y],
            xerr = df[errx].values if errx else None,
            yerr=  df[erry].values if erry else None,
            fmt="o",
            label=legend if usar_leyenda else None
        )
    else:
        ax.plot(
            df[x],
            df[y],
            "o",
            label=legend if usar_leyenda else None
        )

    if usar_titulo:
        ax.set_title(titulo)

    if usar_ejes:
        ax.set_xlabel(f'{xlabel}')
        ax.set_ylabel(f'{ylabel}')

    if usar_leyenda:
        ax.legend()

    if usar_grid:
        ax.grid()

    if ajuste:
        ax.plot(df[x], cfit, label=fr"$y=({m:.2f}\pm{sm:.2f})x+({b:.2f}\pm{sb:.2f})$")
    
    with col2:
        col2.pyplot(fig)
