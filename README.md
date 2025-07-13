
---

## ✅ Part 1: Edge AI Prototype

- Built a TensorFlow/Keras image classification model to detect recyclable items.
- Achieved **93.75% accuracy** after training on 4 classes (glass, metal, paper, plastic).
- Converted the model to **TensorFlow Lite (`.tflite`)** for deployment on Edge devices.

🔹 Technologies: `TensorFlow`, `Keras`, `Jupyter`, `TFLite`

---

## ✅ Part 2: AI-Driven IoT – Smart Agriculture

- Designed a smart farming system using AI + IoT.
- Proposed sensors like: **Soil Moisture**, **Temperature**, **Humidity**, **Light**, and **pH**.
- Used supervised ML model (e.g., Random Forest) to predict crop yield.
- Included a **data flow diagram** showing sensor → AI → farmer dashboard.

---

## ✅ Part 3: Ethics in Personalized Medicine

- Identified biases in genomic data (e.g., underrepresentation of ethnic groups in TCGA).
- Suggested fairness strategies: **diverse datasets**, **bias checks**, and **transparency**.
- Wrote a 300-word ethical reflection as part of the submission.

---

## ✅ Part 4: Human-AI Collaboration

- Reflected on how AI and humans can work together in fields like healthcare and design.
- Emphasized that AI should **augment**, not replace, human judgment.
- Included real-world examples like AI-assisted radiology and creative tools.

---

## 📌 Notes

- All images in `recyclables_dataset/` were downloaded from free sources like [Pexels.com](https://www.pexels.com/).
- Model was trained and evaluated in `Jupyter Notebook`.
  Smart Agriculture Data Flow Diagram

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    SMART AGRICULTURE SYSTEM                                              │
│                                    Data Flow Architecture                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   IoT SENSORS   │    │  WEATHER DATA   │    │  CAMERA SYSTEM  │    │  MANUAL INPUT   │
│                 │    │                 │    │                 │    │                 │
│ • Soil Moisture │    │ • Temperature   │    │ • Crop Health   │    │ • Farmer Notes  │
│ • pH Levels     │    │ • Humidity      │    │ • Disease Det.  │    │ • Observations  │
│ • Nutrients     │    │ • Wind Speed    │    │ • Growth Stage  │    │ • Local Events  │
│ • Temperature   │    │ • Precipitation │    │ • Pest Presence │    │ • Market Data   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │                      │
          └──────────────────────┼──────────────────────┼──────────────────────┘
                                 │                      │
                                 ▼                      ▼
                    ┌─────────────────────────────────────────────────┐
                    │              EDGE GATEWAY                       │
                    │                                                 │
                    │ • Data Aggregation                              │
                    │ • Signal Processing                             │
                    │ • Initial Filtering                             │
                    │ • Local Storage                                 │
                    └─────────────────┬───────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────────────┐
                    │            LOCAL AI PROCESSING                 │
                    │                                                 │
                    │ • Real-time Image Analysis                     │
                    │ • Sensor Data Interpretation                   │
                    │ • Immediate Decision Making                    │
                    │ • Alert Generation                             │
                    └─────────────────┬───────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────────────┐
                    │              CONTROL SYSTEMS                   │
                    │                                                 │
                    │ • Irrigation Control                            │
                    │ • Fertilizer Dispensing                        │
                    │ • Pest Management                              │
                    │ • Climate Control                              │
                    └─────────────────┬───────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────────────┐
                    │              CLOUD ANALYTICS                   │
                    │                                                 │
                    │ • Historical Data Analysis                     │
                    │ • Predictive Modeling                          │
                    │ • Weather Pattern Analysis                     │
                    │ • Yield Optimization                           │
                    └─────────────────┬───────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────────────┐
                    │              DECISION SUPPORT                  │
                    │                                                 │
                    │ • Recommendations                              │
                    │ • Risk Assessment                              │
                    │ • Resource Planning                            │
                    │ • Market Analysis                              │
                    └─────────────────┬───────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────────────┐
                    │              OUTPUT SYSTEMS                    │
                    │                                                 │
                    │ • Mobile App Notifications                     │
                    │ • Dashboard Reports                            │
                    │ • Automated Actions                            │
                    │ • Data Export                                 │
                    └─────────────────────────────────────────────────┘

DATA FLOW SUMMARY:
1. Sensors collect real-time environmental data
2. Edge gateway aggregates and processes data locally
3. AI algorithms analyze data for immediate decisions
4. Control systems execute automated actions
5. Cloud analytics provide long-term insights
6. Decision support system offers recommendations
7. Output systems deliver results to users

BENEFITS:
• Real-time monitoring and response
• Reduced water and resource waste
• Increased crop yield and quality
• Predictive maintenance 

 
