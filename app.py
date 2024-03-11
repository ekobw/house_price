import streamlit as st
import pickle
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def main():
    # stc.html(html_temp)
    # st.title("House Price Analytics (Jabodetabek)")
    st.markdown("""
            <h1 style="text-align: center; font-size: 42px; color: #023047; font-weight: bold">
                House Price Analytics (Jabodetabek)</h1>""", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center'>This dashboard was created for the Capstone Project Tetris Batch 4 from DQLab</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.image("house_price.jpg")

        menu = ["Dashboard", "Prediction"]
        choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Dashboard":
        st.markdown("""
            <h1 style="text-align: center; font-size: 36px; color: #023047; font-weight: bold">
                Business Understanding</h1>""", unsafe_allow_html=True)
        # st.header("Business Understanding")
        st.markdown("""Jakarta is the capital of Indonesia and the center of the economy.
                    Many residents from villages and cities outside Jakarta have moved and settled in areas around Jakarta,
                    because they want to work or earn a living in the capital city area.
                    As a result, the population around the Jakarta area is increasing.
                    So the need for housing will of course also increase.""")

        st.markdown("""This project was created to analyze house prices in the Jakarta area
                    and several surrounding areas, such as **Bogor**, **Depok**, **Bekasi**, **Tangerang** and **Tangerang Selatan**.""")

        st.markdown("""
            <h1 style="text-align: center; font-size: 36px; color: #023047; font-weight: bold">
                Dataset Overview</h1>""", unsafe_allow_html=True)
        # st.header("Dataset Overview")

        url = "https://raw.githubusercontent.com/ekobw/house_price_prediction/main/data/clean_house_price.csv"
        df = pd.read_csv(url)
        top_10_rows = df.head(10)
        st.table(top_10_rows)

        text1a = """
                - The original dataset consists of a total of 9,000 rows (entries) and 6 columns of variables. After cleaning and transformation, the amount of clean data becomes 7,252 rows (entries) and 5 columns of variables.
                - The dataset contains house information data from 6 regions, namely **Jakarta**, **Bogor**, **Depok**, **Tangerang**, **Bekasi** and **Tangerang Selatan**.
                - The independent variables consists of **kamar_tidur**, **luas_bangunan_m2**, **luas_tanah_m2**, and **lokasi** which contain information about the house specifications.
                - The dependent variable is **harga**, which informs the selling price of the house.
                """

        text1b = """
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
                Pearson Correlation and Correlation Matrix shows that **luas_bangunan_m2** and **luas_tanah_m2** variables have a stronger relationship with **harga** variable than the **kamar_tidur** variable. \
                It can be concluded that houses that have a larger building area or land area tend to have higher prices than houses that have many bedrooms.
                """

        conclusion = """
                1. The data distribution shows that the graph of all numerical variables is right-skewed, which means the range of values is quite wide, but the data spread out tends to be more numerous at low values. So the specifications and price of the houses are still quite affordable for sale.
                2. The variables that have significant impact on the selling price of a house are **building area** and **land area**. The larger the building area or land area, the higher the house prices.
                3. Houses located in the Jakarta area have much higher prices compared to houses located outside Jakarta. This makes sense because, as the capital and economic center, so many residents work or earn a living in Jakarta. By having a house in the Jakarta area, they no longer need to spend a lot of time commuting every day to go home for their activities of working or earning a living. That's why the price of houses in the Jakarta area is more expensive than in other areas.
                """

        st.markdown("""
            <p style="font-size: 16px; font-weight: bold">Dataset Description:</p>
            """, unsafe_allow_html=True)
        st.markdown(text1a)

        st.markdown("""
            <p style="font-size: 16px; font-weight: bold">Features:</p>
            """, unsafe_allow_html=True)
        st.markdown(text1b)


        # Display the chart title
        st.markdown("""
            <h1 style="text-align: center; font-size: 36px; color: #023047; font-weight: bold">
                Distribution of Data</h1>""", unsafe_allow_html=True)
        # st.title("Distribution of Data")

        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

        # Create histograms for each numeric column
        histograms = []
        for col in numeric_columns:
            # Create histogram with mean and median annotations
            histogram = alt.Chart(df).mark_bar().encode(
                alt.X(col, bin=alt.Bin(maxbins=40)),
                y='count()'
            ).properties(
                width=300,  # decrease width
                height=150,  # decrease height
                title=f'Distribution of {col}'
            )

            # Add mean and median lines
            histogram += alt.Chart(df).mark_rule(color='red').encode(
                x=f'average({col}):Q',
                size=alt.value(2),
                opacity=alt.value(0.7)
            )

            histogram += alt.Chart(df).mark_rule(color='yellow').encode(
                x=f'median({col}):Q',
                size=alt.value(2),
                opacity=alt.value(0.7)
            )

            histograms.append(histogram)

        # Arrange histograms in a grid layout
        histogram_grid = alt.vconcat(*[alt.hconcat(*histograms[i:i+2]) for i in range(0, len(histograms), 2)])

        # Display Altair chart
        st.altair_chart(histogram_grid, use_container_width=True)


        # # Create histograms for each numeric column
        # histograms = []
        # for col in numeric_columns:
        #     histogram = alt.Chart(df).mark_bar().encode(
        #         alt.X(col, bin=alt.Bin(maxbins=40)),
        #         y='count()'
        #     ).properties(
        #         width=300,
        #         height=150,
        #         title=f'Distribution of {col}'
        #     )
        #     histograms.append(histogram)

        # # Arrange histograms in a 2x2 grid layout
        # histograms_grid = []
        # for i in range(0, len(histograms), 2):
        #     row = alt.hconcat(*histograms[i:i+2])
        #     histograms_grid.append(row)
        # histogram_grid = alt.vconcat(*histograms_grid)

        # # Display Altair chart
        # st.altair_chart(histogram_grid, use_container_width=True)

        st.markdown(text2)


        # Display the chart title and explanation
        st.markdown("""
            <h1 style="text-align: center; font-size: 36px; color: #023047; font-weight: bold">
                Number of Houses Being Sold per City</h1>""", unsafe_allow_html=True)
        # st.title("Number of Houses Being Sold per City")
        st.markdown("<p style='text-align: center'>This chart visualizes the distribution of houses across different cities.</p>", unsafe_allow_html=True)
        # st.write("This chart visualizes the distribution of houses across different cities.")

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
        st.markdown("""
            <h1 style="text-align: center; font-size: 36px; color: #023047; font-weight: bold">
                Average House Price per City</h1>""", unsafe_allow_html=True)
        # st.title("Average House Price per City")
        st.markdown("<p style='text-align: center'>This chart visualizes the average sale price of houses across different cities.</p>", unsafe_allow_html=True)
        # st.write("This chart visualizes the average sale price of houses across different cities.")

        # Compute the average house price per city
        mean_prices = df.groupby('kota')['harga'].mean().sort_values()

        # Create Altair chart
        chart = alt.Chart(mean_prices.reset_index()).mark_bar().encode(
            x=alt.X('harga:Q', title='Average Price', axis=alt.Axis(format=',d')),
            y=alt.Y('kota:N', title='City', sort='-x')
        ).properties(
            width=500,
            height=300,
        )

        # Display Altair chart
        st.altair_chart(chart, use_container_width=True)

        st.markdown(text4)


        # Display the chart title and explanation
        st.markdown("""
            <h1 style="text-align: center; font-size: 36px; color: #023047; font-weight: bold">
                Average Building Area per City</h1>""", unsafe_allow_html=True)
        # st.title("Average Building Area per City")
        st.markdown("<p style='text-align: center'>This chart visualizes the average building area of houses across different cities.</p>", unsafe_allow_html=True)
        # st.write("This chart visualizes the average building area of houses across different cities.")

        # Compute the average house price per city
        mean_prices = df.groupby('kota')['luas_bangunan_m2'].mean().sort_values()

        # Create Altair chart
        chart = alt.Chart(mean_prices.reset_index()).mark_bar().encode(
            x=alt.X('luas_bangunan_m2:Q', title='Average Building Area (m2)', axis=alt.Axis(format=',d')),
            y=alt.Y('kota:N', title='City', sort='-x')
        ).properties(
            width=500,
            height=300,
        )

        # Display Altair chart
        st.altair_chart(chart, use_container_width=True)

        #st.markdown(text5)


        # Display the chart title and explanation
        st.markdown("""
            <h1 style="text-align: center; font-size: 36px; color: #023047; font-weight: bold">
                Average Land Area per City</h1>""", unsafe_allow_html=True)
        # st.title("Average Land Area per City")
        st.markdown("<p style='text-align: center'>This chart visualizes the average land area of houses across different cities.</p>", unsafe_allow_html=True)
        # st.write("This chart visualizes the average land area of houses across different cities.")

        # Compute the average house price per city
        mean_prices = df.groupby('kota')['luas_tanah_m2'].mean().sort_values()

        # Create Altair chart
        chart = alt.Chart(mean_prices.reset_index()).mark_bar().encode(
            x=alt.X('luas_tanah_m2:Q', title='Average Land Area (m2)', axis=alt.Axis(format=',d')),
            y=alt.Y('kota:N', title='City', sort='-x')
        ).properties(
            width=500,
            height=300,
        )

        # Display Altair chart
        st.altair_chart(chart, use_container_width=True)

        #st.markdown(text6)


        # # Create visualization function
        # def visualize(df):
        #     # Create scatter plot using Altair
        #     scatter_plot = alt.Chart(df).mark_circle(size=60).encode(
        #         x='kamar_tidur:Q',
        #         y='harga:Q',
        #         color=alt.value('#fb5607'),  # Set color to orange
        #         tooltip=['kamar_tidur', 'harga']  # Show tooltip with bedroom count and price
        #     ).properties(
        #         width=700,
        #         height=400,
        #         title='Correlation between count of bedrooms and house price'
        #     )

        #     # Add regression line with specified color
        #     regression_line = scatter_plot.transform_regression(
        #         'kamar_tidur', 'harga'
        #     ).mark_line(color='#0775fb')  # Set color to blue

        #     # Combine scatter plot and regression line
        #     chart = scatter_plot + regression_line

        #     # Calculate Pearson correlation coefficient
        #     pearson_corr, _ = pearsonr(df['kamar_tidur'], df['harga'])
        #     st.write("Pearson correlation coefficient:", pearson_corr)

        #     # Display scatter plot with regression line
        #     st.altair_chart(chart, use_container_width=True)

        # # Display the chart title and explanation
        # st.title("Average House Price per City")
        # st.write("This chart visualizes the average sale price of houses across different cities.")

        # # Call the visualize function
        # visualize(df)


        # Display the chart title and explanation
        st.markdown("""
            <h1 style="text-align: center; font-size: 36px; color: #023047; font-weight: bold">
                Pearson Correlation</h1>""", unsafe_allow_html=True)
        # st.title("Pearson Correlation")


        # Create visualization function
        def visualize(df, x_col, y_col, title):
            # Create scatter plot using Seaborn
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(data=df, x=x_col, y=y_col, color='#0775fb',
                        scatter_kws={'edgecolor': 'white'}, line_kws={"color": "#fb5607"}, ax=ax)  # Set regression line color
            ax.set_title(title, fontsize=17)
            ax.set_xlabel(x_col.capitalize(), fontsize=14)
            ax.set_ylabel('House Price', fontsize=14)

            # Calculate Pearson correlation coefficient
            pearson_corr, _ = pearsonr(df[x_col], df['harga'])
            st.write("Pearson correlation coefficient:", pearson_corr)

            # Display scatter plot
            st.pyplot(fig)

        # Layout columns for displaying visuals side by side
        col1, col2, col3 = st.columns(3)

        # Call the visualize function for each pair of variables
        with col1:
            visualize(df, 'kamar_tidur', 'harga', 'Correlation between count of bedrooms and house price')

        with col2:
            visualize(df, 'luas_bangunan_m2', 'harga', 'Correlation between building area and house price')

        with col3:
            visualize(df, 'luas_tanah_m2', 'harga', 'Correlation between land area and house price')


        # # Create visualization function
        # def visualize(df):
        #     # Create scatter plot using Seaborn
        #     fig, ax = plt.subplots(figsize=(10, 6))
        #     sns.regplot(data=df, x='kamar_tidur', y='harga', color='#0775fb',
        #                 scatter_kws={'edgecolor': 'white'}, line_kws={"color": "#fb5607"}, ax=ax)  # Set regression line color
        #     ax.set_title('Correlation between count of bedrooms and house price', fontsize=17)
        #     ax.set_xlabel('Count of Bedrooms', fontsize=14)
        #     ax.set_ylabel('House Price', fontsize=14)

        #     # Calculate Pearson correlation coefficient
        #     pearson_corr, _ = pearsonr(df['kamar_tidur'], df['harga'])
        #     st.write("Pearson correlation coefficient:", pearson_corr)

        #     # Display scatter plot
        #     st.pyplot(fig)

        # # Call the visualize function
        # visualize(df)


        #         # Create visualization function
        # def visualize(df):
        #     # Create scatter plot using Seaborn
        #     fig, ax = plt.subplots(figsize=(10, 6))
        #     sns.regplot(data=df, x='luas_bangunan_m2', y='harga', color='#0775fb',
        #                 scatter_kws={'edgecolor': 'white'}, line_kws={"color": "#fb5607"}, ax=ax)  # Set regression line color
        #     ax.set_title('Correlation between building area and house price', fontsize=17)
        #     ax.set_xlabel('Building Area (m2)', fontsize=14)
        #     ax.set_ylabel('House Price', fontsize=14)

        #     # Calculate Pearson correlation coefficient
        #     pearson_corr, _ = pearsonr(df['luas_bangunan_m2'], df['harga'])
        #     st.write("Pearson correlation coefficient:", pearson_corr)

        #     # Display scatter plot
        #     st.pyplot(fig)

        # # Call the visualize function
        # visualize(df)


        #         # Create visualization function
        # def visualize(df):
        #     # Create scatter plot using Seaborn
        #     fig, ax = plt.subplots(figsize=(10, 6))
        #     sns.regplot(data=df, x='luas_tanah_m2', y='harga', color='#0775fb',
        #                 scatter_kws={'edgecolor': 'white'}, line_kws={"color": "#fb5607"}, ax=ax)  # Set regression line color
        #     ax.set_title('Correlation between land area and house price', fontsize=17)
        #     ax.set_xlabel('Land Area (m2)', fontsize=14)
        #     ax.set_ylabel('House Price', fontsize=14)

        #     # Calculate Pearson correlation coefficient
        #     pearson_corr, _ = pearsonr(df['luas_tanah_m2'], df['harga'])
        #     st.write("Pearson correlation coefficient:", pearson_corr)

        #     # Display scatter plot
        #     st.pyplot(fig)

        # # Call the visualize function
        # visualize(df)


        # Display the chart title
        st.markdown("""
            <h1 style="text-align: center; font-size: 36px; color: #023047; font-weight: bold">
                Correlation Matrix of Numeric Variables</h1>""", unsafe_allow_html=True)
        # st.title("Correlation Matrix of Numeric Variables")

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
            height=500,
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


        st.markdown("""
            <h1 style="text-align: center; font-size: 36px; color: #023047; font-weight: bold">
                Conclusion</h1>""", unsafe_allow_html=True)
        # st.header("Conclusion")
        st.markdown(conclusion)

        st.caption('Copyright :copyright: 2024 by Eko B.W.: https://www.linkedin.com/in/eko-bw')

    elif choice == "Prediction":
        st.markdown("""
            <h1 style="text-align: center; font-size: 36px; color: #023047; font-weight: bold">
                House Price Prediction Application</h1>""", unsafe_allow_html=True)
        # st.header("House Price Prediction Application")
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
    # st.markdown("""
    # <p style="font-size: 16px; font-weight: bold">House Price Predictions</p>
    # """, unsafe_allow_html=True)

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
