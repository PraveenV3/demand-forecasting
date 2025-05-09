import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations(output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)

    # Load processed weekly data
    weekly_path = "data/processed/weekly_waste.csv"
    weekly_df = pd.read_csv(weekly_path)

    # Plot weekly waste volume by area over time
    plt.figure(figsize=(12, 6))
    for area in weekly_df['area'].unique():
        area_data = weekly_df[weekly_df['area'] == area]
        plt.plot(area_data['week'], area_data['net_weight_kg'], label=area)
    plt.title("Weekly Waste Volume by Area")
    plt.xlabel("Week")
    plt.ylabel("Net Weight (kg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weekly_waste_volume_by_area.png"))
    plt.close()

    # Load processed seasonal data
    seasonal_path = "data/processed/seasonal_waste.csv"
    seasonal_df = pd.read_csv(seasonal_path)

    # Plot seasonal waste volume by area
    plt.figure(figsize=(12, 6))
    sns.barplot(data=seasonal_df, x='season', y='net_weight_kg', hue='area')
    plt.title("Seasonal Waste Volume by Area")
    plt.xlabel("Season")
    plt.ylabel("Net Weight (kg)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "seasonal_waste_volume_by_area.png"))
    plt.close()

    print(f"Visualizations saved to folder: {output_dir}")

if __name__ == "__main__":
    create_visualizations()
