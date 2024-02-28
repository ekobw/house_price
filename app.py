import streamlit as st
import streamlit.components.v1 as stc
import pickle
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

with open('./data/final_model.pkl','rb') as file:
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

        url = "https://raw.githubusercontent.com/ekobw/house_price_prediction/main/data/clean_house_price.csv"
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
                - **kota** : Name of the city where the house is being sold
                - **harga** : The price of the house being sold
                """

        text2 = """
                From the histogram chart above, we can see that the graph is right-skewed. \
                This means that the range of data values is quite wide, but the data distribution is not evenly distributed. \
                Most of the data has a low value, meaning that the most sold houses have specifications and prices that are still quite affordable.
                """

        text3 = """
                From the bar chart above, we can see that the number of houses being sold for each region is more or less the same. \
                Likewise for the Jakarta area, if it is accumulated, the total is around 1300 houses for sale for the entire Jakarta area.
                """

        text4 = """
                From the bar chart above, we can see that the average price of houses sold in the Jakarta area is higher than in areas outside Jakarta. \
                Almost all areas of Jakarta are in the top position, except East Jakarta which is below South Tangerang. \
                This may occur due to the unequal amount of data in the two cities, where data for the East Jakarta area is less than for South Tangerang area.
                """

        text5 = """
                Correlation Matrix shows that luas_bangunan_m2 and luas_tanah_m2 variables have a stronger relationship than the kamar_tidur variable. \
                It can be concluded that houses that have a larger building area or land area tend to have higher prices than houses that have many bedrooms.
                """

        st.markdown("""
            <p style="font-size: 16px; font-weight: bold">Dataset Description</p>
            """, unsafe_allow_html=True)

        st.markdown(text1)

        # Display the chart title
        st.title("Distribution of Data")

        # Create the figure without labels
        fig, ax = plt.subplots()
        df.hist(bins=20, color='skyblue', edgecolor='black', ax=ax)

        # Add labels and title separately below the chart
        ax.set_xlabel('Values')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Data')

        plt.tight_layout()  # Adjust spacing

        # Show the plot using st.pyplot
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


        # Display the chart title and explanation
        st.title("Average House Price per City")
        st.write("This chart visualizes the average sale price of houses across different cities.")

        # Sort and filter data for better presentation (optional)
        mean_prices = df.groupby('kota')['harga'].mean().sort_values(ascending=True)

        # Create the bar chart within a Streamlit container
        with st.container():
            plt.figure(figsize=(8, 6))
            bars2 = plt.barh(mean_prices.index, mean_prices, color='lightgreen')
            plt.title('Average House Price per City')
            plt.ylabel('City')
            plt.xlabel('Average Price')

            # Add labels to bars
            plt.bar_label(bars2, fontsize=10)

            plt.tight_layout()
            st.pyplot(plt)

        st.markdown(text4)


        # Display the chart title
        st.title("Correlation Matrix of Numeric Variables")

        # Filter out columns with object data type
        numeric_df = df.select_dtypes(include=['float64', 'int64'])

        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr()

        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create heatmap
        sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, ax=ax)
        ax.set_title('Correlation Matrix of Numeric Variables')

        # Display heatmap
        st.pyplot(fig)

        st.markdown(text5)


    elif choice == "Machine Learning":
        st.header("Prediction Model")
        run_ml_app()


def run_ml_app():

    st.markdown("""
    <p style="font-size: 16px; font-weight: bold">Insert Data</p>
    """, unsafe_allow_html=True)

    left, right = st.columns((2,2))
    kota = left.selectbox('Location',
                            ('Jakarta Pusat', 'Jakarta Utara', 'Jakarta Barat',
                             'Jakarta Selatan', 'Jakarta Timur', 'Bogor', 'Depok',
                             'Bekasi', 'Tangerang', 'Tangerang Selatan'))
    kamar_tidur = left.number_input('Number of Bedrooms', 0, 50)
    luas_bangunan_m2 = right.number_input('Building Area (m2)', 0, 5000)
    luas_tanah_m2 = right.number_input('Land Area (m2)', 0, 10000)

    button = st.button('Predict House Prices')

    #if button is clicked (ketika button dipencet)
    if button:
        try:
            # Preprocess user input
            if kota == 'Jakarta Pusat':
                kota_encoded = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif kota == 'Jakarta Utara':
                kota_encoded = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            elif kota == 'Jakarta Barat':
                kota_encoded = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            elif kota == 'Jakarta Selatan':
                kota_encoded = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            elif kota == 'Jakarta Timur':
                kota_encoded = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            elif kota == 'Bogor':
                kota_encoded = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            elif kota == 'Depok':
                kota_encoded = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            elif kota == 'Bekasi':
                kota_encoded = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
            elif kota == 'Tangerang':
                kota_encoded = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            elif kota == 'Tangerang Selatan':
                kota_encoded = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

            # Convert kota_encoded list to array
            kota_encoded_array = np.array(kota_encoded)

            # Combine all input features to a 2D array
            input_data = np.array([[kota_encoded_array + [kamar_tidur, luas_bangunan_m2, luas_tanah_m2]]])

            # Load the trained model
            model = joblib.load('./data/final_model.pkl')

            # Making prediction
            prediction = model.predict(input_data)

            # Format hasil
            result = f"Harga Rumah Diperkirakan: Rp {prediction[0]:,.2f}"

            st.success(result)
        except Exception as e:
            st.error(f"Terjadi Kesalahan: {e}")

if __name__ == "__main__":
    run_ml_app()


# def run_ml_app():

#     st.markdown("""
#     <p style="font-size: 16px; font-weight: bold">Insert Data</p>
#     """, unsafe_allow_html=True)

#     left, right = st.columns((2,2))
#     kota = left.selectbox('Location',
#                             ('Jakarta Pusat', 'Jakarta Utara', 'Jakarta Barat', 'Jakarta Selatan', 'Jakarta Timur', 'Bogor', 'Depok', 'Bekasi', 'Tangerang', 'Tangerang Selatan'))
#     kamar_tidur = left.number_input('Number of Bedrooms', 0, 50)
#     luas_bangunan_m2 = right.number_input('Building Area (m2)', 0, 5000)
#     luas_tanah_m2 = right.number_input('Land Area (m2)', 0, 10000)

#     button = st.button('Predict House Prices')

#     #if button is clicked (ketika button dipencet)
#     if button:
#         #make prediction
#         result = predict(kota, kamar_tidur, luas_bangunan_m2, luas_tanah_m2)
#         st.write('Predicted House Price:', result)

# def encode_kota(kota):
#     if kota == 'Jakarta Pusat':
#         kota_encoded = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     elif kota == 'Jakarta Utara':
#         kota_encoded = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#     elif kota == 'Jakarta Barat':
#         kota_encoded = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#     elif kota == 'Jakarta Selatan':
#         kota_encoded = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#     elif kota == 'Jakarta Timur':
#         kota_encoded = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
#     elif kota == 'Bogor':
#         kota_encoded = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
#     elif kota == 'Depok':
#         kota_encoded = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
#     elif kota == 'Bekasi':
#         kota_encoded = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
#     elif kota == 'Tangerang':
#         kota_encoded = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
#     elif kota == 'Tangerang Selatan':
#         kota_encoded = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

# def predict(kota, kamar_tidur, luas_bangunan_m2, luas_tanah_m2):
#     # Encode 'kota' feature
#     kota_encoded = encode_kota(kota)

#     # Combine all input features to a 2D array
#     input_data = np.array([[kota_encoded + [kamar_tidur, luas_bangunan_m2, luas_tanah_m2]]])

#     # Load the trained model
#     model = joblib.load('./data/final_model.pkl')

#     # Making prediction
#     prediction = model.predict([[kota_encoded, kamar_tidur, luas_bangunan_m2, luas_tanah_m2]])

#     return prediction

# if __name__ == "__main__":
#     run_ml_app()



# def run_ml_app():

#     st.markdown("""
#     <p style="font-size: 16px; font-weight: bold">Insert Data</p>
#     """, unsafe_allow_html=True)

#     left, right = st.columns((2,2))
#     kota = left.selectbox('Location',
#                             ('Jakarta Pusat', 'Jakarta Utara', 'Jakarta Barat', 'Jakarta Selatan', 'Jakarta Timur', 'Bogor', 'Depok', 'Bekasi', 'Tangerang', 'Tangerang Selatan'))
#     kamar_tidur = left.number_input('Number of Bedrooms', 0, 50)
#     luas_bangunan_m2 = right.number_input('Building Area (m2)', 0, 5000)
#     luas_tanah_m2 = right.number_input('Land Area (m2)', 0, 10000)

#     button = st.button('Predict House Prices')

#     #if button is clicked (ketika button dipencet)
#     if button:
#         #make prediction
#         result = predict(kota, kamar_tidur, luas_bangunan_m2, luas_tanah_m2)
#         if result == 'Eligible':
#             st.success(f'You have {result} from the loan')
#         else:
#             st.warning(f'You have {result} for the loan')

# def predict(kota, kamar_tidur, luas_bangunan_m2, luas_tanah_m2):
#     #processing user input
#     gen = 0 if gender == 'Male' else 1
#     cre = 0 if has_credit_card == 'No' else 1

#     #Making prediction
#     prediction = Final_Model.predict([[kota, kamar_tidur, luas_bangunan_m2, luas_tanah_m2]])
#     result = prediction

#     return result

# if __name__ == "__main__":
#     main()