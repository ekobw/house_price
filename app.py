import streamlit as st
import pickle
import pandas as pd
import numpy as np
import altair as alt

def main():
    # stc.html(html_temp)
    st.title("House Price Prediction App")
    st.markdown("""
            <p style="font-size: 38px; color: #023047;font-weight: bold">House Price Analytics (Jabodetabek)</p>
            """, unsafe_allow_html=True)
    st.markdown("This dashboard was created for the Capstone Project Tetris Batch 4 from DQLab")

    with st.sidebar:
        st.image("house_price.jpg")

        menu = ["Dashboard", "Prediction"]
        choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Dashboard":
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

                - The original dataset consists of a total of 9,000 rows (entries) and 6 columns of variables. After cleaning and transformation, the amount of clean data becomes 7,252 rows (entries) and 5 columns of variables.
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
                Likewise for the Jakarta area, if it is accumulated, the total is around 1200 houses for sale for the entire Jakarta area.
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

    elif choice == "Prediction":
        st.header("Prediction Model")
        run_ml_app()

def run_ml_app():
    # Load model
    with open('./data/final_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load scaler
    with open('./data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Define function to encode kota
    def encode_kota(kota):
        if kota == 'Jakarta Pusat':
            return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif kota == 'Jakarta Utara':
            return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif kota == 'Jakarta Barat':
            return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif kota == 'Jakarta Selatan':
            return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif kota == 'Jakarta Timur':
            return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif kota == 'Bogor':
            return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif kota == 'Depok':
            return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif kota == 'Bekasi':
            return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif kota == 'Tangerang':
            return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif kota == 'Tangerang Selatan':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    # Initialize Streamlit app
    st.markdown("This house price prediction application uses a Machine Learning model that has been trained using previous datasets.")
    st.markdown("""
    <p style="font-size: 16px; font-weight: bold">House Price Predictions</p>
    """, unsafe_allow_html=True)

    # Create sidebar for user input
    left, right = st.columns((2,2))
    kota = left.selectbox('Location',
                        ('Jakarta Pusat', 'Jakarta Utara', 'Jakarta Barat',
                        'Jakarta Selatan', 'Jakarta Timur', 'Bogor', 'Depok',
                        'Bekasi', 'Tangerang', 'Tangerang Selatan'))
    kamar_tidur = left.number_input('Number of Bedrooms', 0, 50)
    luas_bangunan_m2 = right.number_input('Building Area (m2)', 0, 5000)
    luas_tanah_m2 = right.number_input('Land Area (m2)', 0, 10000)

    # Predict button
    button = st.button('Price Prediction')

    # Make prediction and show result
    if button:
        try:
            # Preprocess user input
            kota_encoded = encode_kota(kota)
            kota_features = np.array(kota_encoded)
            other_features = np.array([kamar_tidur, luas_bangunan_m2, luas_tanah_m2])

            # Combine all features
            input_data = np.concatenate([other_features, kota_features])

            # Reshape input data
            input_data_reshaped = input_data.reshape(1, -1)

            # Transform input data using the loaded scaler
            input_data_scaled = scaler.transform(input_data_reshaped)

            # Make prediction
            prediction = model.predict(input_data_scaled)

            # Format result
            result = f"Estimated House Prices: Rp {prediction[0]:,.2f}"
            st.success(result)
        except Exception as e:
            st.error(f"Sorry, something wrong: {e}")

    st.caption('Copyright :copyright: 2024 by Eko B.W.: https://www.linkedin.com/in/eko-bw')

# Call the function to run the ML app
if __name__ == '__main__':
     main()
