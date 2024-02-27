import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

with open('random_forest_model.pkl','rb') as file:
    Final_Model = pickle.load(file)

def main():
    # stc.html(html_temp)
    # st.title("House Price Prediction App")
    st.markdown("""
            <p style="font-size: 44px; color: #023047;font-weight: bold">House Price Prediction App</p>
            """, unsafe_allow_html=True)
    st.markdown("This application was created for the Capstone Project Tetris Batch 4 from DQLab")

    with st.sidebar:
        st.image("house_price.jpg")

        menu = ["Overview","Machine Learning"]
        choice = st.sidebar.selectbox("Menu", menu)


    if choice == "Overview":
        st.header("Overview")
        st.markdown("House Price Prediction App utilize machine learning to predict house prices based on house specifications and locations. \
                    This allows users to find out the price range of the house they want to sell, or find out the price of the house they are looking for so they can adjust it to their budget.")

        st.markdown("""
            <p style="font-size: 16px; font-weight: bold">Dataset Overview</p>
            """, unsafe_allow_html=True)

        url = "https://raw.githubusercontent.com/ekobw/house_price_prediction/main/data/house_price_clean.csv"
        df = pd.read_csv(url)
        top_10_rows = df.head(10)
        st.table(top_10_rows)

        text1 = """

                - This dataset consists of a total of 8,298 rows (entries) and contains 5 columns of variable.
                - The dataset contains house information data from 6 regions, namely **Jakarta**, **Bogor**, **Depok**, **Tangerang**, **Bekasi** and **Tangerang Selatan**.
                - The independent variables consist of **kamar_tidur**, **luas_bangunan_m2**, **luas_tanah_m2**, and **lokasi** which contain information about the house specifications.
                - The dependent variable is **harga**, which informs the selling price of the house.

                Features:

                - **kamar_tidur** : Number of bedrooms
                - **luas_bangunan_m2** : Building area of the house in square meters
                - **luas_tanah_m2** : Land area of the house in square meters
                - **kota** : Name of the city where the house is being sale
                - **harga** : The price of the house being sale
                """

        text2 = """
                From the histogram chart above, we can see that the graph is right-skewed. \
                This means that the range of data values is quite wide, but the data distribution is not evenly distributed. \
                Most of the data has a low value, meaning that the most sale houses have specifications and prices that are still quite affordable.
                """


        text3 = """
                From the bar chart above, it can be seen that the number of houses sold for each region is more or less the same. \
                Likewise for the Jakarta area, if it is accumulated, the total is around 1300 houses for sale for the entire Jakarta area.
                """


        text4 = """
                1. **Korelasi antara Umur (Age) dan Exited:**
                - Korelasi positif menunjukkan bahwa ada hubungan yang moderat antara usia nasabah dan kecenderungan untuk keluar dari layanan.
                - Ini dapat diartikan bahwa semakin tua seseorang, semakin cenderung mereka bertahan dalam layanan.

                2. **Korelasi antara Jenis Kelamin (Gender) dan Exited:**
                - Korelasi negatif menunjukkan bahwa terdapat hubungan cukup negatif antara jenis kelamin (laki-laki) dan kecenderungan untuk keluar dari layanan.
                - Hal ini dapat diartikan bahwa nasabah perempuan mungkin cenderung lebih loyal terhadap layanan dibandingkan dengan nasabah laki-laki.

                3. **Korelasi antara Kepemilikan Kartu Kredit (HasCrCard) dan Exited:**
                - Korelasi negatif menunjukkan bahwa kepemilikan kartu kredit memiliki pengaruh cukup negatif terhadap kecenderungan keluar dari layanan.
                - Artinya, nasabah yang memiliki kartu kredit cenderung lebih setia terhadap layanan.

                4. **Korelasi antara Skor Kredit (CreditScore) dan Exited:**
                - Korelasi negatif menunjukkan bahwa terdapat hubungan yang kurang kuat antara skor kredit dan kecenderungan keluar dari layanan.
                - Hal ini mungkin menandakan bahwa nasabah dengan skor kredit yang lebih tinggi memiliki kecenderungan yang sedikit lebih rendah untuk keluar dari layanan.

                5. **Korelasi antara Estimasi Pendapatan (EstimatedSalary) dan Exited:**
                - Korelasi positif yang sangat lemah menunjukkan bahwa tidak ada korelasi yang signifikan antara estimasi pendapatan dan kecenderungan keluar dari layanan.
                - Dengan kata lain, estimasi pendapatan tidak menjadi faktor utama yang mempengaruhi keputusan nasabah untuk keluar dari layanan.

                5. **Korelasi antara Kepemilikan Kartu Kredit (HasCrCard) dan Jenis Kelamin (Gender):**
                - Korelasi positif menunjukkan bahwa ada hubungan positif yang kurang kuat antara kepemilikan kartu kredit dan jenis kelamin laki-laki.
                - Artinya, laki-laki mungkin sedikit lebih mungkin memiliki kartu kredit.
                """

        st.markdown("""
            <p style="font-size: 16px; font-weight: bold">Dataset Description</p>
            """, unsafe_allow_html=True)

        st.markdown(text1)

        # Memilih hanya kolom-kolom numerik dari dataframe
        numeric_cols = df.select_dtypes(include=['int', 'float']).columns

        # Membuat histogram untuk setiap kolom numerik
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.hist(df[col], bins=20, color='skyblue', edgecolor='black')
            ax.set_xlabel(col)
            ax.set_ylabel('Frekuensi')
            ax.set_title(f'Distribusi {col}')
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown(text2)


        # Display the chart title and explanation
        st.title("Number of Houses for Sale per City")
        st.write("This chart visualizes the distribution of houses across different cities.")

        # Sort and filter data for better visualization (optional)
        value_counts = df['kota'].value_counts().sort_values(ascending=True)

        # # Create the bar chart within a Streamlit container
        with st.container():
            plt.figure(figsize=(8, 6))
            bars1 = plt.barh(value_counts.index, value_counts, color='skyblue')
            plt.title('Number of Houses for Sale per City')
            plt.ylabel('City')
            plt.xlabel('Number of Houses')

            # Add labels to bars
            plt.bar_label(bars1, fontsize=10)

            plt.tight_layout()
            st.pyplot(plt)

        st.markdown(text3)


        # Display the chart title
        st.title("Distribution of Data")

        # Create the histogram within a Streamlit container
        with st.container():
            plt.figure(figsize=(14, 8))
            df.hist(bins=20, color='skyblue', edgecolor='black')
            plt.title('Distribution of Data')
            plt.xlabel('Values')
            plt.ylabel('Count')
            plt.tight_layout()
            st.pyplot(plt)

        st.markdown("""
            <p style="font-size: 16px; font-weight: bold">Mengatasi Imbalance Dataset</p>
            """, unsafe_allow_html=True)
        #st.image("output2.png")
        st.markdown(text4)

    elif choice == "Machine Learning":
        st.header("Prediction Model")
        run_ml_app()

def run_ml_app():

    st.markdown("""
    <p style="font-size: 16px; font-weight: bold">Insert Data</p>
    """, unsafe_allow_html=True)

    left, right = st.columns((2,2))
    kota = left.selectbox('Location',
                            ('Jakarta Pusat', 'Jakarta Utara', 'Jakarta Barat', 'Jakarta Selatan', 'Jakarta Timur', 'Bogor', 'Depok', 'Bekasi', 'Tangerang', 'Tangerang Selatan'))
    kamar_tidur = left.number_input('Number of Bedrooms', 0, 50)
    luas_bangunan_m2 = right.number_input('Building Area (m2)', 0, 5000)
    luas_tanah_m2 = right.number_input('Land Area (m2)', 0, 10000)

    button = st.button('Predict House Prices')

    #if button is clicked (ketika button dipencet)
    if button:
        #make prediction
        result = predict(kota, kamar_tidur, luas_bangunan_m2, luas_tanah_m2)
        if result == 'Eligible':
            st.success(f'You have {result} from the loan')
        else:
            st.warning(f'You have {result} for the loan')

def predict(kota, kamar_tidur, luas_bangunan_m2, luas_tanah_m2):
    #processing user input
    gen = 0 if gender == 'Male' else 1
    cre = 0 if has_credit_card == 'No' else 1

    #Making prediction
    prediction = Final_Model.predict([[kota, kamar_tidur, luas_bangunan_m2, luas_tanah_m2]])
    result = prediction

    return result

if __name__ == "__main__":
    main()