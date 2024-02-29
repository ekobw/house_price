import streamlit as st
import streamlit.components.v1 as stc
import pickle
import joblib
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import RobustScaler

# with open('./data/final_model.pkl','rb') as file:
#     Final_Model = pickle.load(file)

def main():
    # stc.html(html_temp)
    # st.title("House Price Prediction App")
    st.markdown("""
            <p style="font-size: 38px; color: #023047;font-weight: bold">House Price Analytics (Jabodetabek)</p>
            """, unsafe_allow_html=True)
    st.markdown("This dashboard was created for the Capstone Project Tetris Batch 4 from DQLab")

    with st.sidebar:
        st.image("house_price.jpg")

        menu = ["Overview", "Machine Learning"]
        choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Overview":
        st.header("Overview")
        st.markdown("This is a dashboard for analyzing the prices of houses sold in the Jakarta, Bogor, Depok, Tangerang, Bekasi and Tangerang Selatan areas.")

        st.markdown("""
            <p style="font-size: 16px; font-weight: bold">Dataset Overview</p>
            """, unsafe_allow_html=True)

        url = "https://raw.githubusercontent.com/ekobw/house_price_prediction/main/data/clean_house_price.csv"
        df = pd.read_csv(url)
        top_10_rows = df.head(10)
        st.table(top_10_rows)

        text1 = """

                - This dataset consists of a total of 7,252 rows (entries) and contains 5 columns of variable.
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
                Correlation Matrix shows that **luas_bangunan_m2** and **luas_tanah_m2** variables have a stronger relationship with **harga** variable than the **kamar_tidur** variable. \
                It can be concluded that houses that have a larger building area or land area tend to have higher prices than houses that have many bedrooms.
                """

        st.markdown("""
            <p style="font-size: 16px; font-weight: bold">Dataset Description</p>
            """, unsafe_allow_html=True)

        st.markdown(text1)

        # Display the chart title
        st.title("Distribution of Data")

        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

        # Create histograms for each numeric column
        histograms = []
        for col in numeric_columns:
            histogram = alt.Chart(df).mark_bar().encode(
                alt.X(col, bin=alt.Bin(maxbins=20)),
                y='count()'
            ).properties(
                width=300,
                height=200,
                title=f'Distribution of {col}'
            )
            histograms.append(histogram)

        # Arrange histograms in a grid layout
        histogram_grid = alt.vconcat(*histograms)

        # Display Altair chart
        st.altair_chart(histogram_grid, use_container_width=True)

        st.markdown(text2)


        # Display the chart title and explanation
        st.title("Number of Houses Being Sold per City")
        st.write("This chart visualizes the distribution of houses across different cities.")

        # Count the number of houses per city
        house_counts = df['kota'].value_counts().reset_index()
        house_counts.columns = ['kota', 'jumlah']

        # Sort the DataFrame by 'jumlah' column in descending order
        house_counts = house_counts.sort_values(by='jumlah', ascending=False)

        # Create Altair chart
        chart = alt.Chart(house_counts).mark_bar().encode(
            x=alt.X('jumlah:Q', title='Number of Houses Being Sold'),
            y=alt.Y('kota:N', title='City', sort='-x')  # Sort the bars by 'jumlah' in descending order
        ).properties(
            width=500,
            height=300
        )

        # Add labels to bars
        text = chart.mark_text(
            align='left',
            baseline='middle',
            dx=3,  # Nudge text to right side of bar
            color='black'  # Set text color
        ).encode(
            text='jumlah:Q'  # Use 'jumlah' as text
        )

        # Combine chart and text
        chart = (chart + text).interactive()

        # Display Altair chart
        st.altair_chart(chart, use_container_width=True)

        st.markdown(text3)


        # Display the chart title and explanation
        st.title("Average House Price per City")
        st.write("This chart visualizes the average sale price of houses across different cities.")

        # Compute the average house price per city
        mean_prices = df.groupby('kota')['harga'].mean().sort_values()

        # Create Altair chart
        chart = alt.Chart(mean_prices.reset_index()).mark_bar().encode(
            x=alt.X('harga:Q', title='Average Price'),
            y=alt.Y('kota:N', title='City', sort='-x')
        ).properties(
            width=500,
            height=300,
        )

        # Display Altair chart
        st.altair_chart(chart, use_container_width=True)

        st.markdown(text4)


        # Display the chart title
        st.title("Correlation Matrix of Numeric Variables")

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['float64', 'int64'])

        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr().reset_index().rename(columns={'index': 'variable1'})

        # Melt correlation matrix
        melted_df = pd.melt(correlation_matrix, id_vars='variable1', var_name='variable2', value_name='correlation')

        # Create heatmap using Altair
        heatmap = alt.Chart(melted_df).mark_rect().encode(
            x='variable1:N',
            y='variable2:N',
            color='correlation:Q',
            tooltip=['variable1', 'variable2', 'correlation']
        ).properties(
            width=500,
            height=400,
        )

        # Add text on heatmap
        text = heatmap.mark_text(baseline='middle').encode(
            text=alt.Text('correlation:Q', format='.2f'),
            color=alt.condition(
                alt.datum.correlation > 0.5,
                alt.value('white'),
                alt.value('black')
            )
        )

        # Combine heatmap and text layers
        chart = (heatmap + text)

        # Display the chart
        st.altair_chart(chart, use_container_width=True)

        st.markdown(text5)

        st.caption('Copyright :copyright: 2024 by Eko B.W.: https://www.linkedin.com/in/eko-bw')

    elif choice == "Machine Learning":
        st.header("Prediction Model")
        run_ml_app()


def run_ml_app():
    # Load model pickle
    model = pickle.load(open('./data/final_model.pkl', 'rb'))

    # Load encoding and scaling objects
    encoding_object = pickle.load(open('./data/encoding_object.pkl', 'rb'))
    scaling_object = pickle.load(open('./data/scaling_object.pkl', 'rb'))

    # Function to encode city name
    def encode_city(city):
        # Create a mapping dictionary
        mapping = {
            'Jakarta Pusat': 'kota_jakarta_pusat',
            'Jakarta Selatan': 'kota_jakarta_selatan',
            'Jakarta Barat': 'kota_jakarta_barat',
            'Jakarta Utara': 'kota_jakarta_utara',
            'Jakarta Timur': 'kota_jakarta_timur',
            'Bogor': 'kota_bogor',
            'Depok': 'kota_depok',
            'Bekasi': 'kota_bekasi',
            'Tangerang': 'kota_tangerang',
            'Tangerang Selatan': 'kota_tangerang_selatan'
        }

        # Map the selected city to the corresponding column name
        encoded_city = mapping.get(city)

        return encoded_city

    # Function to scale input values
    def scale_values(kamar_tidur, luas_bangunan_m2, luas_tanah_m2):
        scaled_values = scaling_object.transform([[kamar_tidur, luas_bangunan_m2, luas_tanah_m2]])
        return scaled_values[0]

    # Main function to run the Streamlit app
    city = st.selectbox('Pilih Nama Kota', ['Jakarta Pusat', 'Jakarta Selatan', 'Jakarta Barat',
                                            'Jakarta Utara', 'Jakarta Timur', 'Bogor', 'Depok',
                                            'Bekasi', 'Tangerang', 'Tangerang Selatan'])

    # Textbox for input values
    kamar_tidur = st.text_input('Kamar Tidur')
    luas_bangunan_m2 = st.text_input('Luas Bangunan (m2)')
    luas_tanah_m2 = st.text_input('Luas Tanah (m2)')

    # Button to predict house price
    if st.button('Prediksi Harga Rumah'):
        # Encode city
        encoded_city = encode_city(city)

        # Scale input values
        scaled_values = scale_values(int(kamar_tidur), int(luas_bangunan_m2), int(luas_tanah_m2))

        # Combine encoded city and scaled values
        features = [0] * len(encoding_object.feature_names_in_)  # Initialize features with zeros
        city_index = encoding_object.feature_names_in_.index(encoded_city)
        features[city_index] = 1  # Set the value for the encoded city
        features.extend(scaled_values)  # Add scaled values

        # Predict house price
        prediction = model.predict([features])
        st.write('Prediksi Harga Rumah:', prediction[0])




if __name__ == '__main__':
    main()


#================================================================================================================

# import streamlit as st
# import pandas as pd
# import pickle
# from sklearn.preprocessing import RobustScaler

# # Load the model and scaler
# @st.cache(hash_funcs={builtins.dict: my_hash_func})
# def load_model():
#     with open('./data/final_model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     return model

# model, scaler = load_model()

# def encode_city(city):
#     cities = ['Jakarta Pusat', 'Jakarta Utara', 'Jakarta Barat', 'Jakarta Selatan', 'Jakarta Timur',
#               'Bogor', 'Depok', 'Bekasi', 'Tangerang', 'Tangerang Selatan']
#     encoded_city = [0] * len(cities)
#     if city in cities:
#         index = cities.index(city)
#         encoded_city[index] = 1
#     return encoded_city

# # Function to predict house price
# def predict_price(encoded_city, bedrooms_scaled, building_area_scaled, land_area_scaled):
#     features = np.concatenate([encoded_city, [bedrooms_scaled, building_area_scaled, land_area_scaled]])
#     prediction = model.predict(features.reshape(1, -1))  # Reshape if model expects single-sample input
#     return prediction[0]

# # Preprocess input data (scaling removed)
# def preprocess_input(city, bedrooms, building_area, land_area):
#     # Encode city
#     encoded_city = encode_city(city)

#     # Combine input features
#     features = [[bedrooms, building_area, land_area]]  # Hanya fitur numerik yang diikutsertakan

#     # Scale input features
#     scaler = RobustScaler()
#     features_scaled = scaler.fit_transform(features)

#     # Flatten the scaled features
#     bedrooms_scaled, building_area_scaled, land_area_scaled = features_scaled[0]

#     return encoded_city, bedrooms, building_area, land_area

# # Streamlit app
# def main():
#     st.title("House Price Prediction")

#     # Dropdown for city selection
#     city = st.selectbox("Select city", ["City 1", "City 2", "City 3", "City 4", "City 5", "City 6", "City 7", "City 8", "City 9"])

#     # Textboxes for input
#     bedrooms = st.number_input("Number of bedrooms", min_value=1, step=1)
#     building_area = st.number_input("Building area (m^2)", min_value=0, step=1)
#     land_area = st.number_input("Land area (m^2)", min_value=0, step=1)

#     # Preprocess input data
#     encoded_city, bedrooms, building_area, land_area = preprocess_input(city, bedrooms, building_area, land_area)

#     # Scale features using loaded scaler
#     features = scaler.transform([[bedrooms, building_area, land_area]])

#     # Predict house price
#     if st.button("Predict"):
#         price_prediction = predict_price(encoded_city, bedrooms_scaled, building_area_scaled, land_area_scaled)
#         st.success(f"Predicted house price: {price_prediction}")

# if __name__ == "__main__":
#     main()

#=================================================================================================================

# import streamlit as st
# import pandas as pd
# import pickle
# from sklearn.preprocessing import RobustScaler

# # Load the model
# @st.cache(allow_output_mutation=True)
# def load_model():
#     with open('./data/final_model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     return model

# model = load_model()

# def encode_city(city):
#     cities = ['Jakarta Pusat', 'Jakarta Utara', 'Jakarta Barat', 'Jakarta Selatan', 'Jakarta Timur',
#               'Bogor', 'Depok', 'Bekasi', 'Tangerang', 'Tangerang Selatan']
#     encoded_city = [0] * len(cities)
#     if city in cities:
#         index = cities.index(city)
#         encoded_city[index] = 1
#     return encoded_city

# # Function to predict house price
# def predict_price(encoded_city, bedrooms, building_area, land_area):
#     features = [encoded_city + [bedrooms, building_area, land_area]]  # Menggabungkan fitur-fitur menjadi satu array
#     prediction = model.predict(features)
#     return prediction[0]

# # Function to preprocess input data
# def preprocess_input(city, bedrooms, building_area, land_area):
#     # Encode city
#     encoded_city = encode_city(city)

#     # Combine input features
#     #features = [[bedrooms, building_area, land_area]]  # Hanya fitur numerik yang diikutsertakan
#     features = [encoded_city + [bedrooms, building_area, land_area]]

#     # Scale input features
#     scaler = RobustScaler()
#     features_scaled = scaler.fit_transform(features)

#     # Check the length of features_scaled[0]
#     if len(features_scaled[0]) != 3:
#         raise ValueError("Unexpected number of scaled features")

#     # Flatten the scaled features
#     bedrooms_scaled, building_area_scaled, land_area_scaled = features_scaled[0]

#     return encoded_city, bedrooms_scaled, building_area_scaled, land_area_scaled

# # Streamlit app
# def main():
#     st.title("House Price Prediction")

#     # Dropdown for city selection
#     city = st.selectbox("Select city", ["City 1", "City 2", "City 3", "City 4", "City 5", "City 6", "City 7", "City 8", "City 9"])

#     # Textboxes for input
#     bedrooms = st.number_input("Number of bedrooms", min_value=1, step=1)
#     building_area = st.number_input("Building area (m^2)", min_value=0, step=1)
#     land_area = st.number_input("Land area (m^2)", min_value=0, step=1)

#     # Preprocess input data
#     encoded_city, bedrooms_scaled, building_area_scaled, land_area_scaled = preprocess_input(city, bedrooms, building_area, land_area)

#     # Predict house price
#     if st.button("Predict"):
#         price_prediction = predict_price(encoded_city, bedrooms_scaled, building_area_scaled, land_area_scaled)
#         st.success(f"Predicted house price: {price_prediction}")

# if __name__ == "__main__":
#     main()

