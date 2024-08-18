from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import networkx as nx
from scipy.spatial import ConvexHull

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request.'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'})
    
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)
    
    try:
        edges = extract_edges_from_image(image_path)
        G = create_graph_from_edges(edges)
        if len(G.nodes()) >= 3:
            trend_points = extract_trend(G)
            summary = summarize_trend(trend_points)
            return jsonify({'summary': summary})
        else:
            return jsonify({'error': 'Not enough points in the graph to form a trend line.'})
    except Exception as e:
        return jsonify({'error': str(e)})

def extract_edges_from_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    return edges

def create_graph_from_edges(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    G = nx.Graph()
    for contour in contours:
        for i in range(len(contour)):
            for j in range(i + 1, len(contour)):
                point1 = tuple(contour[i][0])
                point2 = tuple(contour[j][0])
                if point1 != point2:
                    G.add_edge(point1, point2)
    return G

def extract_trend(G):
    positions = np.array([node for node in G.nodes()])
    if len(positions) < 3:
        raise ValueError("Not enough points to form a convex hull. Ensure the image contains sufficient features.")
    hull = ConvexHull(positions)
    trend_points = positions[hull.vertices]
    trend_points = trend_points[trend_points[:, 0].argsort()]
    return trend_points

def summarize_trend(trend_points):
    slopes = []
    for i in range(1, len(trend_points)):
        dx = trend_points[i][0] - trend_points[i - 1][0]
        dy = trend_points[i][1] - trend_points[i - 1][1]
        if dx != 0:
            slopes.append(dy / dx)
    
    avg_slope = np.mean(slopes)
    direction = "upward" if avg_slope > 0 else "downward" if avg_slope < 0 else "horizontal"
    
    summary = f"Overall Trend: The trend is generally {direction}.\n"
    summary += f"Average Slope: {avg_slope:.2f}\n"
    
    significant_changes = []
    for i, slope in enumerate(slopes):
        if abs(slope) > 1.5 * abs(avg_slope):
            change = "increase" if slope > 0 else "decrease"
            significant_changes.append(f"Significant {change} between points {i} and {i + 1}.")
    
    if significant_changes:
        summary += "Significant Changes:\n" + "\n".join(significant_changes)
    else:
        summary += "No significant changes detected."
    
    return summary

if __name__ == '__main__':
    app.run(debug=True)
