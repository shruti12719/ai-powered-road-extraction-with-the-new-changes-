# AI-powered-disaster-response-and-emergency-route-planning-platform.
AI  powered disaster response system that automatically extract the road networks from high resolution satellite images using deep learning  
# 🛰️ RouteResilience-AI
### AI-Powered Disaster Response & Emergency Route Planning using Satellite Imagery

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-ff4b4b)
![OpenStreetMap](https://img.shields.io/badge/OpenStreetMap-GIS-green)
![NetworkX](https://img.shields.io/badge/NetworkX-GraphAnalysis-orange)
![License](https://img.shields.io/badge/License-MIT-blue)

</p>

---

## 📌 Overview

**RouteResilience-AI** is an end-to-end AI-powered decision support system that transforms satellite imagery into actionable disaster intelligence.

Unlike traditional road extraction systems that stop after generating segmentation masks, RouteResilience-AI converts extracted roads into a connected transportation network, evaluates infrastructure resilience, simulates disaster scenarios, and recommends optimal emergency evacuation and rescue routes through an interactive GIS dashboard.

The platform is designed to assist emergency responders, disaster management agencies, urban planners, and government organizations in making faster and more informed decisions during natural disasters.

---

## 🎯 Problem Statement

Natural disasters such as floods, landslides, cyclones, earthquakes, and urban fires frequently damage transportation networks.

Emergency responders often face challenges such as:

- Blocked or destroyed roads
- Isolated communities
- Incomplete road information
- Delayed rescue operations
- Lack of real-time decision support

RouteResilience-AI addresses these challenges by automatically extracting road networks from satellite imagery and generating resilient emergency routes even under disrupted network conditions.

---

# ✨ Key Features

## 🛰️ AI-Based Road Extraction

- Satellite image upload
- Deep Learning road segmentation
- Automatic road mask generation
- Morphological refinement
- Skeletonization

---

## 🌍 Interactive GIS Dashboard

- Google Maps-like interface
- Interactive road visualization
- Satellite basemap
- Automatic city navigation
- Search by place name
- Live map interaction

---

## 📍 Intelligent Location Search

Search locations naturally instead of entering coordinates.

Examples:

- Hospitals
- Fire Stations
- Police Headquarters
- Relief Camps
- Airports
- Railway Stations
- Roads
- Localities

The application automatically:

- Converts place names into geographic coordinates
- Finds the nearest road network
- Snaps locations onto the graph

---

## 🛣️ Road Network Generation

The extracted road mask is transformed into a graph representation.

Features include:

- Node generation
- Edge creation
- Road connectivity analysis
- Graph optimization
- Automatic topology repair

---

## 🚨 Disaster Simulation

Simulate multiple disaster scenarios:

- Flood
- Landslide
- Earthquake
- Bridge Collapse
- Major Accident
- Road Construction

Road failures are automatically reflected in the routing graph.

---

## 🚑 Emergency Route Planning

Generate

- Normal Route
- Emergency Alternate Route
- Safe Route Recommendation

The system automatically avoids damaged or blocked roads whenever an alternate path exists.

---

## 📊 Infrastructure Analytics

Analyze

- Critical Roads
- Critical Junctions
- Network Connectivity
- Graph Density
- Betweenness Centrality
- Road Importance Ranking
- Disaster Impact Score

---

## 📈 Decision Support

The platform automatically generates:

- AI Summary
- Delay Analysis
- Connectivity Score
- Recovery Suggestions
- Restoration Priorities

---

## 📄 Report Generation

Export

- Road Mask
- Route Analysis
- Disaster Report
- Network Statistics
- PDF Summary

---
## output 

<img width="1367" height="502" alt="image" src="https://github.com/user-attachments/assets/3c996653-32a1-47c3-a1fc-0cd981c4a394" />

# 🏗️ System Architecture

```text
Satellite Image
        │
        ▼
Road Extraction (Deep Learning)
        │
        ▼
Road Mask Refinement
        │
        ▼
Skeletonization
        │
        ▼
Road Graph Generation
        │
        ▼
Topology Healing
        │
        ▼
Critical Infrastructure Detection
        │
        ▼
Disaster Simulation
        │
        ▼
Emergency Route Planning
        │
        ▼
Interactive GIS Dashboard
        │
        ▼
Decision Support Report
```

---

# 🧠 AI Pipeline

```
Satellite Image
        │
        ▼
Deep Learning Segmentation
        │
        ▼
Road Mask
        │
        ▼
Morphological Processing
        │
        ▼
Skeleton Extraction
        │
        ▼
Graph Construction
        │
        ▼
Graph Healing
        │
        ▼
Network Analysis
        │
        ▼
Disaster Simulation
        │
        ▼
Emergency Route Generation
        │
        ▼
Interactive Dashboard
```

---

# 💻 Technology Stack

## Artificial Intelligence

- PyTorch
- Segmentation Models PyTorch
- U-Net
- DeepLabV3+
- SegFormer

---

## Computer Vision

- OpenCV
- scikit-image
- NumPy

---

## Graph Intelligence

- NetworkX
- Dijkstra Algorithm
- Connected Components
- Betweenness Centrality
- Graph Healing

---

## GIS

- Folium
- OpenStreetMap
- Nominatim
- Leaflet

---

## Dashboard

- Streamlit
- Plotly
- Pandas

---

# 🚀 Getting Started

## Clone Repository

```bash
git clone https://github.com/yourusername/RouteResilience-AI.git

cd RouteResilience-AI
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

# 📸 Application Workflow

1. Upload satellite image

2. AI extracts road network

3. Road graph is generated

4. Select city

5. Search source location

6. Search destination

7. Select disaster type

8. Generate emergency route

9. Analyze resilience

10. Download report

---

# 📊 Future Improvements

- Real-time satellite imagery
- Live weather integration
- Multi-city support
- Traffic-aware routing
- ISRO Bhuvan integration
- Drone imagery support
- Multi-agent rescue planning
- Dynamic road updates
- Mobile application
- Cloud deployment

---

# 🌍 Applications

- Disaster Response
- Smart Cities
- Emergency Logistics
- Flood Management
- Military Route Planning
- Humanitarian Relief
- Urban Planning
- Infrastructure Monitoring

---



