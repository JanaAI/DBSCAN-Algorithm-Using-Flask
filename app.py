from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import os
import io
import base64

app = Flask(__name__)

app.config["UPLOAD_FOLDER1"] = "static/excel"

@app.route("/", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        upload_excel = request.files['upload_excel']
        if upload_excel.filename != '':
            file_path = os.path.join(app.config["UPLOAD_FOLDER1"], upload_excel.filename)
            upload_excel.save(file_path)
            data = pd.read_excel(upload_excel)
            columns = data.columns.tolist()
            return render_template("Excel.html", columns=columns, data=data.to_html(index=False).replace('<th>', '<th style="text-align:center">'))
    return render_template("UploadExcel.html")

@app.route("/calculate_epsilon", methods=['POST'])
def calculate_epsilon():
    selected_features = request.form.getlist('features')
    
    # Load the uploaded data from the saved file
    file_path = os.path.join(app.config["UPLOAD_FOLDER1"], "sample_1.xlsx")
    data = pd.read_excel(file_path)
    
    # Filter data by selected features
    data_selected = data[selected_features]
    
    # Determine the optimal epsilon
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import DBSCAN
    import numpy as np
    import matplotlib.pyplot as plt

    # Calculate distances for k-distance graph
    neighbors = NearestNeighbors(n_neighbors=4)
    neighbors_fit = neighbors.fit(data_selected)
    distances, indices = neighbors_fit.kneighbors(data_selected)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 3]  # Use the distance to the 3rd nearest neighbor

    # Find the knee point
    def find_knee_point(distances):
        # Simple heuristic for finding the knee point
        gradients = np.diff(distances)
        knee_point_index = np.argmax(gradients)
        return knee_point_index, distances[knee_point_index]
    
    knee_point_index, knee_point_value = find_knee_point(distances)
    
    # Plot k-distance graph with knee point
    plt.figure(figsize=(10, 6))
    plt.plot(distances, label='k-distance')
    plt.axvline(x=knee_point_index, color='r', linestyle='--', label='Knee Point')
    plt.scatter(knee_point_index, knee_point_value, color='red')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel('k-distance')
    plt.title('K-Distance Graph with Knee Point')
    plt.legend()
    plt.savefig(os.path.join(app.config["UPLOAD_FOLDER1"], "k_distance_graph.png"))
    plt.close()

    # Find the optimal epsilon
    optimal_epsilon = distances[knee_point_index]  # Use the distance at the knee point
    min_pts = max(4, int(np.log(data_selected.shape[0])))  # Use heuristic for min_pts

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=optimal_epsilon, min_samples=min_pts)
    clusters = dbscan.fit_predict(data_selected)
    
    # Plot clustering results
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_selected.iloc[:, 0], data_selected.iloc[:, 1], c=clusters, cmap='viridis')
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.title('DBSCAN Clustering Results')
    plt.savefig(os.path.join(app.config["UPLOAD_FOLDER1"], "cluster_graph.png"))
    plt.close()

    # Render the result page with epsilon, min_pts values, and graphs
    return render_template("results.html", 
                           epsilon=optimal_epsilon, 
                           min_pts=min_pts, 
                           k_distance_graph_url="/static/excel/k_distance_graph.png",
                           cluster_graph_url="/static/excel/cluster_graph.png")

if __name__ == "__main__":
    app.run(debug=True)
