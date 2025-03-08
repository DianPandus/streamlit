from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, count, avg, month, year, countDistinct, min, max
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

# Set style seaborn
sns.set(style='dark')

# Inisialisasi SparkSession dengan konfigurasi memori
spark = SparkSession.builder \
    .appName("ECommerceDashboard") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Baca file CSV dengan PySpark
file_path = "all_data.csv"  # Ganti dengan path file CSV Anda
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Konversi kolom tanggal ke timestamp
df = df.withColumn("order_purchase_timestamp", col("order_purchase_timestamp").cast("timestamp"))

# Ambil tanggal minimum dan maksimum
min_date = df.select(min("order_purchase_timestamp")).collect()[0][0]
max_date = df.select(max("order_purchase_timestamp")).collect()[0][0]

# Sidebar untuk filter data
st.sidebar.header('Filter Data')
start_date = st.sidebar.date_input('Start Date', min_date)
end_date = st.sidebar.date_input('End Date', max_date)

# Filter data berdasarkan tanggal
filtered_df = df.filter(
    (col("order_purchase_timestamp") >= start_date) & 
    (col("order_purchase_timestamp") <= end_date)
)

# Header dashboard
st.title('E-Commerce Dashboard')
st.markdown("""
Dashboard ini menampilkan berbagai analisis terkait penjualan, pelanggan, dan produk dari data e-commerce.
""")

# Metrik utama
st.header('Metrik Utama')
col1, col2, col3 = st.columns(3)
with col1:
    total_orders = filtered_df.select(countDistinct("order_id")).collect()[0][0]
    st.metric("Total Orders", value=total_orders)
with col2:
    total_revenue = filtered_df.select(sum("payment_value")).collect()[0][0]
    st.metric("Total Revenue", value=format_currency(total_revenue, "R$", locale='id_ID'))
with col3:
    avg_order_value = filtered_df.select(avg("payment_value")).collect()[0][0]
    st.metric("Average Order Value", value=format_currency(avg_order_value, "R$", locale='id_ID'))

# Distribusi Harga Produk
st.header('Distribusi Harga Produk')
sampled_df = filtered_df.sample(fraction=0.1, seed=42)  # Ambil 10% sampel data
price_data = sampled_df.select("price").toPandas()
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(price_data['price'], bins=30, kde=True, color='skyblue')
avg_price = price_data['price'].mean()
plt.axvline(avg_price, color='red', linestyle='--', label=f'Rata-rata: {avg_price:.2f}')
plt.title('Distribusi Harga Produk')
plt.xlabel('Harga')
plt.ylabel('Frekuensi')
plt.legend()
st.pyplot(fig)

# Top 10 Kategori Produk dengan Harga Tertinggi
st.header('Top 10 Kategori Produk dengan Harga Tertinggi')
average_price_by_category = filtered_df.groupBy("product_category_name") \
    .agg(avg("price").alias("avg_price")) \
    .orderBy("avg_price", ascending=False) \
    .limit(10) \
    .toPandas()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='avg_price', y='product_category_name', data=average_price_by_category, palette='viridis')
plt.title('Top 10 Kategori Produk dengan Harga Tertinggi')
plt.xlabel('Rata-rata Harga')
plt.ylabel('Kategori Produk')
st.pyplot(fig)

# Top 10 Kategori Produk dengan Penjualan Tertinggi
st.header('Top 10 Kategori Produk dengan Penjualan Tertinggi')
total_sales_by_category = filtered_df.groupBy("product_category_name") \
    .agg(sum("order_item_id").alias("total_sales")) \
    .orderBy("total_sales", ascending=False) \
    .limit(10) \
    .toPandas()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='total_sales', y='product_category_name', data=total_sales_by_category, palette='viridis')
plt.title('Top 10 Kategori Produk dengan Penjualan Tertinggi')
plt.xlabel('Total Penjualan')
plt.ylabel('Kategori Produk')
st.pyplot(fig)

# Tren Revenue Bulanan
st.header('Tren Revenue Bulanan')
revenue_by_month = filtered_df.withColumn("order_month", month("order_purchase_timestamp")) \
    .withColumn("order_year", year("order_purchase_timestamp")) \
    .groupBy("order_year", "order_month") \
    .agg(sum("payment_value").alias("total_revenue")) \
    .orderBy("order_year", "order_month") \
    .toPandas()

revenue_by_month['order_month'] = revenue_by_month['order_month'].astype(str)
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x='order_month', y='total_revenue', data=revenue_by_month, marker='o')
plt.title('Total Revenue per Bulan')
plt.xlabel('Bulan')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(fig)

# Pola Musiman Penjualan
st.header('Pola Musiman Penjualan')
seasonal_sales = filtered_df.withColumn("order_month", month("order_purchase_timestamp")) \
    .groupBy("order_month") \
    .agg(count("order_id").alias("total_orders")) \
    .orderBy("order_month") \
    .toPandas()

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='order_month', y='total_orders', data=seasonal_sales, palette='crest')
plt.title('Pola Musiman Penjualan')
plt.xlabel('Bulan')
plt.ylabel('Jumlah Pesanan')
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
st.pyplot(fig)

# Distribusi Metode Pembayaran
st.header('Distribusi Metode Pembayaran')
payment_frequency = filtered_df.groupBy("payment_type") \
    .agg(count("order_id").alias("count")) \
    .toPandas()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='payment_type', y='count', data=payment_frequency, palette='crest')
plt.title('Distribusi Metode Pembayaran')
plt.xlabel('Metode Pembayaran')
plt.ylabel('Jumlah Transaksi')
st.pyplot(fig)

# Footer
st.caption('Copyright © 2023 by Dian Pandu Syahfitra')

# Stop SparkSession
spark.stop()