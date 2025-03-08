import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
import geopandas as gpd

# Set style seaborn
sns.set(style='dark')

# Load data
all_df = pd.read_csv('all_data.csv')

# Konversi kolom tanggal ke datetime
all_df['order_purchase_timestamp'] = pd.to_datetime(all_df['order_purchase_timestamp'])

# Sidebar untuk filter data
st.sidebar.header('Filter Data')
start_date = st.sidebar.date_input('Start Date', all_df['order_purchase_timestamp'].min())
end_date = st.sidebar.date_input('End Date', all_df['order_purchase_timestamp'].max())

# Filter data berdasarkan tanggal
filtered_df = all_df[(all_df['order_purchase_timestamp'] >= pd.to_datetime(start_date)) & 
                     (all_df['order_purchase_timestamp'] <= pd.to_datetime(end_date))]

# Header dashboard
st.title('E-Commerce Dashboard')
st.markdown("""
Dashboard ini menampilkan berbagai analisis terkait penjualan, pelanggan, dan produk dari data e-commerce.
""")

# Metrik utama
st.header('Metrik Utama')
col1, col2, col3 = st.columns(3)
with col1:
    total_orders = filtered_df['order_id'].nunique()
    st.metric("Total Orders", value=total_orders)
with col2:
    total_revenue = filtered_df['payment_value'].sum()
    st.metric("Total Revenue", value=format_currency(total_revenue, "R$", locale='id_ID'))
with col3:
    avg_order_value = filtered_df['payment_value'].mean()
    st.metric("Average Order Value", value=format_currency(avg_order_value, "R$", locale='id_ID'))

# Distribusi Harga Produk
st.header('Distribusi Harga Produk')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(filtered_df['price'], bins=30, kde=True, color='skyblue')
avg_price = filtered_df['price'].mean()
plt.axvline(avg_price, color='red', linestyle='--', label=f'Rata-rata: {avg_price:.2f}')
plt.title('Distribusi Harga Produk')
plt.xlabel('Harga')
plt.ylabel('Frekuensi')
plt.legend()
st.pyplot(fig)

# Top 10 Kategori Produk dengan Harga Tertinggi
st.header('Top 10 Kategori Produk dengan Harga Tertinggi')
average_price_by_category = filtered_df.groupby('product_category_name')['price'].mean().reset_index()
average_price_by_category = average_price_by_category.sort_values(by='price', ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='price', y='product_category_name', data=average_price_by_category.head(10), palette='viridis')
plt.title('Top 10 Kategori Produk dengan Harga Tertinggi')
plt.xlabel('Rata-rata Harga')
plt.ylabel('Kategori Produk')
st.pyplot(fig)

# Top 10 Kategori Produk dengan Penjualan Tertinggi
st.header('Top 10 Kategori Produk dengan Penjualan Tertinggi')
total_sales_by_category = filtered_df.groupby('product_category_name')['order_item_id'].sum().reset_index()
total_sales_by_category = total_sales_by_category.sort_values(by='order_item_id', ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='order_item_id', y='product_category_name', data=total_sales_by_category.head(10), palette='viridis')
plt.title('Top 10 Kategori Produk dengan Penjualan Tertinggi')
plt.xlabel('Total Penjualan')
plt.ylabel('Kategori Produk')
st.pyplot(fig)

# Tren Revenue Bulanan
st.header('Tren Revenue Bulanan')
filtered_df['order_purchase_month'] = filtered_df['order_purchase_timestamp'].dt.to_period('M')
total_revenue_by_month = filtered_df.groupby('order_purchase_month')['payment_value'].sum().reset_index()
total_revenue_by_month['order_purchase_month'] = total_revenue_by_month['order_purchase_month'].astype(str)
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x='order_purchase_month', y='payment_value', data=total_revenue_by_month, marker='o')
plt.title('Total Revenue per Bulan')
plt.xlabel('Bulan')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(fig)

# Pola Musiman Penjualan
st.header('Pola Musiman Penjualan')
filtered_df['order_purchase_month_only'] = filtered_df['order_purchase_timestamp'].dt.month
seasonal_sales = filtered_df.groupby('order_purchase_month_only').size().reset_index(name='total_orders')
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='order_purchase_month_only', y='total_orders', data=seasonal_sales, palette='crest')
plt.title('Pola Musiman Penjualan')
plt.xlabel('Bulan')
plt.ylabel('Jumlah Pesanan')
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
st.pyplot(fig)

# Distribusi Metode Pembayaran
st.header('Distribusi Metode Pembayaran')
payment_frequency = filtered_df['payment_type'].value_counts().reset_index()
payment_frequency.columns = ['payment_type', 'count']
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='payment_type', y='count', data=payment_frequency, palette='crest')
plt.title('Distribusi Metode Pembayaran')
plt.xlabel('Metode Pembayaran')
plt.ylabel('Jumlah Transaksi')
st.pyplot(fig)

# Visualisasi Geospatial
st.header('Distribusi Pelanggan pada Peta Dunia')

# Load dataset geolokasi
geolocation_data = pd.read_csv('geolocation_dataset.csv')

# Load dataset customers
customers_data = pd.read_csv('customers_dataset.csv')

# Gabungkan geolocation_data dengan customers_data berdasarkan customer_zip_code_prefix
merged_data = pd.merge(
    customers_data,
    geolocation_data,
    left_on='customer_zip_code_prefix',
    right_on='geolocation_zip_code_prefix',
    how='left'
)

# Hitung jumlah pelanggan per kota
customer_distribution = merged_data.groupby('customer_city')['customer_unique_id'].nunique().reset_index()

# Gabungkan dengan data geolokasi untuk mendapatkan latitude dan longitude
geolocation_data_city = merged_data[['customer_city', 'geolocation_lat', 'geolocation_lng']].drop_duplicates()
customer_distribution = pd.merge(customer_distribution, geolocation_data_city, on='customer_city')

# Buat GeoDataFrame
gdf = gpd.GeoDataFrame(
    customer_distribution,
    geometry=gpd.points_from_xy(customer_distribution['geolocation_lng'], customer_distribution['geolocation_lat'])
)

# Proyeksi peta dunia
world = gpd.read_file('ne_110m_admin_0_countries.shp').to_crs(epsg=4326)

# Plot peta
fig, ax = plt.subplots(figsize=(20, 10))
world.plot(ax=ax, color='lightgray', edgecolor='black')

# Tambahkan titik pelanggan
ax.scatter(gdf['geolocation_lng'], gdf['geolocation_lat'], color='red', s=10, alpha=0.7)

plt.title('Distribusi Pelanggan pada Peta Dunia', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
st.pyplot(fig)

# Footer
st.caption('Copyright © 2023 by Dian Pandu Syahfitra')