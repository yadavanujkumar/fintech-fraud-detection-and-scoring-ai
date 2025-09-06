# Fintech Fraud Detection and Scoring AI

A comprehensive AI-powered fraud detection and risk scoring platform for financial institutions. This system uses advanced machine learning algorithms, graph analytics, and real-time monitoring to detect fraudulent transactions, assess risk levels, and identify fraud rings.

## 🚀 Features

- **Real-time Fraud Detection**: Detects fraudulent transactions using Isolation Forest and XGBoost algorithms
- **Dynamic Risk Scoring**: Assigns risk scores to users/transactions based on behavior patterns  
- **Graph Analytics**: Uncovers hidden fraud rings and money laundering networks using NetworkX
- **Adaptive Learning**: Continuously learns from new fraud attempts
- **Banking Dashboard**: Interactive web dashboard with alerts, fraud statistics, and explainability
- **API Integration**: RESTful API for FinTech companies to integrate fraud detection
- **Comprehensive Testing**: Full test suite with pytest

## 📁 Project Structure

```
├── src/
│   ├── fraud_detection/     # Core fraud detection engine
│   │   ├── engine.py        # ML models (Isolation Forest, XGBoost)
│   │   ├── data_generator.py # Synthetic data generation
│   │   └── main.py          # Main fraud detection demo
│   ├── risk_scoring/        # Risk scoring algorithms
│   │   └── engine.py        # Dynamic risk assessment
│   ├── graph_analytics/     # Graph-based fraud ring detection
│   │   └── network_analyzer.py # Network analysis & fraud rings
│   ├── dashboard/           # Web dashboard
│   │   └── app.py           # Dash-based interactive dashboard
│   └── api/                 # REST API endpoints
│       └── server.py        # Flask-based API server
├── tests/                   # Unit and integration tests
├── data/                    # Sample datasets (generated)
├── models/                  # Trained ML models (generated)
├── demo.py                  # Complete system demonstration
├── examples.py              # Usage examples
├── api_examples.py          # API usage examples
└── requirements.txt         # Python dependencies
```

## 🔧 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yadavanujkumar/fintech-fraud-detection-and-scoring-ai.git
cd fintech-fraud-detection-and-scoring-ai
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the complete demo:**
```bash
python demo.py
```

## 🎯 Quick Start

### 1. Run Complete System Demo
```bash
python demo.py
```
This demonstrates all features: data generation, fraud detection, risk scoring, network analysis, and performance metrics.

### 2. Start Interactive Dashboard
```bash
python -m src.dashboard.app
```
Then open http://localhost:8050 in your browser to view the interactive dashboard.

### 3. Start API Server
```bash
python -m src.api.server
```
API will be available at http://localhost:5000 with the following endpoints:
- `GET /` - Health check
- `POST /fraud-check` - Single transaction fraud check
- `POST /batch-fraud-check` - Batch transaction processing
- `GET /user-profile/<user_id>` - User risk profile
- `GET /network-analysis` - Network analysis and fraud rings
- `POST /train-models` - Train ML models

### 4. Run Usage Examples
```bash
python examples.py
```
Shows practical examples of using each component.

### 5. Test API Integration
```bash
# Start API server first
python -m src.api.server

# In another terminal, run API examples
python api_examples.py
```

## 🧪 Testing

Run the complete test suite:
```bash
python -m pytest tests/ -v
```

Run specific test modules:
```bash
python -m pytest tests/test_fraud_detection.py -v
python -m pytest tests/test_risk_scoring.py -v
python -m pytest tests/test_graph_analytics.py -v
```

## 📊 Components

### 1. Fraud Detection Engine
- **Isolation Forest**: Unsupervised anomaly detection for identifying unusual transaction patterns
- **XGBoost**: Supervised learning for classification based on labeled fraud data
- **Feature Engineering**: Automatic creation of relevant features from transaction data
- **Model Persistence**: Save and load trained models

```python
from src.fraud_detection.engine import FraudDetectionEngine

engine = FraudDetectionEngine()
engine.train(transaction_data)
predictions = engine.predict(new_transactions)
```

### 2. Risk Scoring System
- **Real-time Assessment**: Calculate risk scores for individual transactions
- **Behavior Analysis**: Track user patterns and detect anomalies
- **Multiple Risk Factors**: Amount, timing, location, velocity, merchant type
- **Dynamic Scoring**: Risk scores update based on user history

```python
from src.risk_scoring.engine import RiskScoringEngine

risk_engine = RiskScoringEngine()
risk_result = risk_engine.calculate_transaction_risk_score(transaction)
print(f"Risk Score: {risk_result['risk_score']}, Level: {risk_result['risk_level']}")
```

### 3. Graph Analytics
- **Network Construction**: Build transaction networks from user-merchant relationships
- **Fraud Ring Detection**: Identify groups of users with suspicious patterns
- **Community Detection**: Find connected components with high fraud rates
- **Network Metrics**: Analyze centrality, density, and connectivity

```python
from src.graph_analytics.network_analyzer import FraudNetworkAnalyzer

analyzer = FraudNetworkAnalyzer()
graph = analyzer.build_transaction_network(transactions)
fraud_rings = analyzer.detect_fraud_rings()
```

### 4. Banking Dashboard
- **Multi-tab Interface**: Overview, Risk Scoring, Network Analysis, Real-time Monitoring
- **Interactive Visualizations**: Charts, graphs, and tables using Plotly/Dash
- **Real-time Updates**: Live monitoring of transactions and alerts
- **Fraud Statistics**: Comprehensive analytics and reporting

### 5. REST API
Complete API for integration with existing systems:

**Single Transaction Check:**
```bash
curl -X POST http://localhost:5000/fraud-check \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_001",
    "user_id": "USER_001", 
    "amount": 1500.0,
    "timestamp": "2023-01-01T10:00:00",
    "location": "New York",
    "merchant_category": "online"
  }'
```

**Response:**
```json
{
  "transaction_id": "TXN_001",
  "fraud_prediction": {
    "isolation_forest": {"is_fraud": false, "anomaly_score": -0.1},
    "xgboost": {"is_fraud": false, "fraud_probability": 0.15}
  },
  "risk_assessment": {
    "risk_score": 25.0,
    "risk_level": "LOW",
    "risk_factors": ["High amount: $1500.0"]
  }
}
```

## 📈 Performance Metrics

The system provides comprehensive performance analytics:

- **Precision**: Accuracy of fraud predictions
- **Recall**: Coverage of actual fraudulent transactions  
- **Risk Distribution**: Breakdown of transactions by risk level
- **Network Metrics**: Graph density, connectivity, centrality measures
- **User Profiles**: Individual risk assessments and behavior patterns

## 🔍 Example Use Cases

### Financial Institution Integration
```python
# Real-time transaction monitoring
for transaction in incoming_transactions:
    result = risk_engine.calculate_transaction_risk_score(transaction)
    if result['risk_score'] >= 70:
        send_alert_to_fraud_team(transaction, result)
        block_transaction(transaction)
```

### Batch Processing for Risk Assessment
```python
# Daily batch processing
daily_transactions = load_daily_transactions()
risk_results = risk_engine.batch_score_transactions(daily_transactions)
high_risk_users = identify_high_risk_users(risk_results)
```

### Fraud Ring Investigation
```python
# Network analysis for fraud investigations
analyzer.build_transaction_network(suspicious_transactions)
fraud_rings = analyzer.detect_fraud_rings(min_ring_size=3, min_fraud_rate=0.6)
generate_investigation_report(fraud_rings)
```

## 🛡️ Security Considerations

- **Data Privacy**: No real financial data is stored; uses synthetic data for demonstrations
- **API Security**: Implement authentication and rate limiting in production
- **Model Security**: Protect trained models from unauthorized access
- **Audit Trail**: Log all fraud detection decisions for regulatory compliance

## 🚧 Production Deployment

For production deployment, consider:

1. **Scalability**: Use distributed computing (Spark) for large-scale processing
2. **Real-time Processing**: Implement Kafka for streaming transactions
3. **Database Integration**: Connect to production transaction databases
4. **Model Management**: Implement MLOps practices for model versioning and deployment
5. **Monitoring**: Add comprehensive logging and monitoring
6. **Security**: Implement proper authentication, authorization, and encryption

## 📚 Documentation

- **API Documentation**: Available at `/docs` when running the API server
- **Code Documentation**: Comprehensive docstrings throughout the codebase
- **Examples**: See `examples.py` and `api_examples.py` for usage patterns
- **Tests**: Test files demonstrate expected behavior and usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🔗 Links

- **Repository**: https://github.com/yadavanujkumar/fintech-fraud-detection-and-scoring-ai
- **Issues**: Report bugs and request features
- **Wiki**: Additional documentation and tutorials

---

**Built with ❤️ for the FinTech community to combat fraud and protect financial systems.**
