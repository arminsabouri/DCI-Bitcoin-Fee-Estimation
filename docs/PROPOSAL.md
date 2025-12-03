# Bitcoin Transaction Fee Estimation: A Two-Stage Model with Impatience-Based Wait Time Prediction

## Executive Summary

This project develops a machine learning system to predict Bitcoin transaction fees and confirmation wait times using a novel two-stage estimation approach. The system leverages mempool data, transaction characteristics, and user behavior signals (CPFP, RBF, respend patterns) to provide accurate fee recommendations. Our key innovation is modeling user "impatience" as a predictive signal for wait times, transforming fee estimation from a purely technical problem into a behavioral prediction challenge.

## 1. Problem Statement

Bitcoin users face a fundamental challenge: determining the optimal transaction fee to achieve their desired confirmation time. The fee market is dynamic, with prices fluctuating based on network congestion, block space demand, and user urgency. Current fee estimation services often provide inaccurate predictions, leading to:

- **Overpayment**: Users pay excessive fees for faster confirmation than needed
- **Underpayment**: Transactions get stuck in the mempool, requiring costly fee-bumping strategies
- **Poor user experience**: Unpredictable confirmation times create uncertainty and frustration

The core challenge is that transaction fees depend on:
1. **Network conditions**: Mempool congestion, block space availability
2. **Transaction characteristics**: Size, weight, fee rate
3. **User behavior**: Urgency levels, impatience signals (CPFP, RBF, respend patterns)
4. **Temporal patterns**: Hour-of-day, day-of-week effects

## 2. Research Questions

1. **Can we predict transaction wait times using network conditions and user behavior signals without knowing the fee rate?** (Stage 1: Wait Time Prediction)
2. **Can predicted wait times be used to estimate optimal fee rates?** (Stage 2: Fee Estimation)
3. **Do user impatience signals (CPFP, RBF, respend) improve wait time predictions?**
4. **What features are most predictive of Bitcoin transaction confirmation delays?**

## 3. Methodology

### 3.1 Two-Stage Estimation Framework

We implement a structural two-stage model inspired by "A Model and Estimation of the Bitcoin Transaction Fee":

**Stage 1: Wait Time Prediction**
```
W_{q,t,i} = f(ρ̂_t, F_t(q), RBF_{it}, CPP_{it}, X_{it}) + ε_{it}
```
Where:
- `W_{q,t,i}`: Wait time (blocks or minutes) for transaction i in epoch t
- `ρ̂_t`: Average mempool congestion during epoch t
- `F_t(q)`: Respend frequency (impatience proxy)
- `RBF_{it}`: Replace-by-Fee indicator
- `CPP_{it}`: Child-Pays-for-Parent indicator
- `X_{it}`: Transaction characteristics (size, weight, etc.)

**Stage 2: Fee Estimation**
```
fee_{it} = α₁ + α₂ρ̂_t + α₃Σ prob(q_t)Ŵ_{qt} + α₄V_{it} + α₅Weight_{it} + controls + ε_{it}
```
Where predicted wait times from Stage 1 feed into the fee estimation model.

### 3.2 Impatience-Based Approach

Our key innovation is modeling user impatience as a predictive signal:

**Multi-Task Learning Architecture:**
- **Primary Task**: Predict wait time `log(wait_time_minutes)`
- **Auxiliary Tasks**: 
  - `P(CPFP within 30min | features)`
  - `P(RBF within 30min | features)`
  - `P(respend within T | features)` (when data available)

**Feature Engineering:**
- **User Urgency Signals**: Fee overpayment, size efficiency, time-of-day patterns
- **Historical Impatience Patterns**: Recent CPFP/RBF rates in similar conditions
- **Market Context**: Congestion severity, fee market volatility, block timing

### 3.3 Data Collection and Processing

**Data Sources:**
- Bitcoin Core RPC: Transaction data, mempool snapshots
- Exchange data: Transaction flow patterns

**Data Pipeline:**
1. **Collection**: Real-time mempool monitoring and transaction tracking
2. **Enrichment**: Merge mempool state, RBF data, respend patterns
3. **Feature Engineering**: 
   - Log transformations for heavy-tailed distributions
   - Temporal encoding (cyclical hour/day features)
   - Fee market position features (percentiles, z-scores)
   - Congestion metrics (mempool size, velocity, queue depth)
4. **Validation**: Time-based splits to prevent temporal leakage

### 3.4 Model Development

**Phase 1: Baseline Models**
- Linear regression (Ridge/Lasso)
- Random Forest
- XGBoost

**Phase 2: Advanced Models**
- Neural Networks (Multi-layer Perceptron)
- Multi-task learning architectures
- Ensemble methods

**Phase 3: Production Models**
- Binary classification (wait time bins)
- Regression models (continuous wait time)
- Fee estimation models using predicted wait times

## 4. Technical Implementation

### 4.1 Architecture

**Components:**
1. **Data Lake** (`src/data_lake/`): SQLite-based storage and processing
2. **Mempool Collector** (`fast_mempool_collector.py`): Real-time data collection
3. **Model Training** (`scripts/`, `src/data_lake/notebooks/`): Jupyter notebooks and Python scripts
4. **Deployment** (`Dockerfile`, `deploy.sh`): Containerized deployment

**Technology Stack:**
- Python 3.8+
- PyTorch (neural networks)
- XGBoost (gradient boosting)
- scikit-learn (baseline models)
- pandas, numpy (data processing)
- SQLite (data storage)

### 4.2 Key Features

**Network State Features:**
- Mempool congestion (`ρ̂_t`): Time-weighted average transactions
- Mempool size and velocity
- Block timing and space availability

**Transaction Features:**
- Size/weight (log-transformed)
- Fee rate (for Stage 2)
- Transaction type indicators

**Behavioral Features:**
- CPFP indicator (Child-Pays-for-Parent)
- RBF indicator (Replace-by-Fee)
- Respend patterns (when available)
- Historical impatience rates

**Temporal Features:**
- Cyclical hour/day encoding
- Time since last block
- Network activity indicators

## 5. Expected Contributions

### 5.1 Academic Contributions
1. **Novel Modeling Approach**: First application of impatience-based multi-task learning to Bitcoin fee estimation
2. **Behavioral Insights**: Quantification of how user impatience affects transaction confirmation
3. **Feature Engineering**: Domain-specific features for Bitcoin fee market dynamics
4. **Empirical Validation**: Comprehensive evaluation of two-stage estimation on real Bitcoin data

### 5.2 Practical Contributions
1. **Improved Fee Estimation**: More accurate predictions than existing services
2. **Open Source Tools**: Reusable codebase for Bitcoin fee analysis
3. **Data Pipeline**: Scalable system for mempool data collection and processing
4. **Production-Ready Models**: Deployable models with proper validation and monitoring

## 6. Evaluation Metrics

### 6.1 Wait Time Prediction
- **R²** (coefficient of determination) on log-transformed wait times
- **Correlation** (Pearson, Spearman) between predicted and actual
- **MAE** (Mean Absolute Error) in minutes
- **Quantile Loss**: Performance at different percentiles (P50, P80, P90)

### 6.2 Fee Estimation
- **R²** on fee rate predictions
- **Cost Savings**: Reduction in overpayment vs. baseline methods
- **Confirmation Rate**: Percentage of transactions confirmed within target time

### 6.3 Success Criteria
- **Minimum**: R² > 0.1, Correlation > 0.3
- **Good**: R² > 0.3, Correlation > 0.5
- **Excellent**: R² > 0.5, Correlation > 0.7

## 7. Project Timeline

### Phase 1: Data Collection & Exploration (Weeks 1-2)
- ✅ Set up mempool data collection pipeline
- ✅ Collect and process historical transaction data
- ✅ Exploratory data analysis and feature engineering

### Phase 2: Wait Time Models (Weeks 3-5)
- ✅ Baseline models (linear, random forest, XGBoost)
- ✅ Neural network architectures
- ✅ Multi-task learning with impatience signals
- ✅ Model evaluation and diagnostics

### Phase 3: Fee Estimation Models (Weeks 6-7)
- ✅ Two-stage fee estimation
- ✅ Integration of wait time predictions
- ✅ Fee recommendation system
- ✅ Performance evaluation

### Phase 4: Deployment & Documentation (Week 8)
- ✅ Production deployment setup
- ✅ API/service development
- ✅ Documentation and code cleanup
- ✅ Final report and presentation

## 8. Current Status

### Completed Work
- ✅ Mempool data collection infrastructure
- ✅ Transaction data processing pipeline
- ✅ Multiple wait time prediction models (NN, XGBoost, classification)
- ✅ Fee estimation models (Phase 3)
- ✅ Feature engineering and data enrichment
- ✅ Model evaluation and diagnostics

### In Progress
- 🔄 Model refinement and hyperparameter tuning
- 🔄 Integration of respend data
- 🔄 Production deployment preparation

### Future Work
- ⏳ Real-time API service
- ⏳ User behavior clustering
- ⏳ Advanced ensemble methods
- ⏳ Cross-validation and robustness testing

## 9. Data and Resources

### Data
- **Mempool Snapshots**: Real-time and historical mempool state
- **Transaction Data**: ~10M+ confirmed transactions with timing data
- **RBF/CPFP Data**: User behavior signals
- **Respend Data**: Custom-scraped respend patterns (when available)

### Infrastructure
- **Storage**: SQLite database (~10GB+)
- **Compute**: Local development + cloud deployment
- **Monitoring**: Mempool tracking and model performance metrics

## 10. Team and Responsibilities

**Kristian Praizner** (Primary Contributor)
- Data collection and pipeline development
- Model development and evaluation
- Feature engineering and analysis
- Documentation and deployment

## 11. References

1. "A Model and Estimation of the Bitcoin Transaction Fee" - Structural model for Bitcoin fees
2. Bitcoin Core Documentation - RPC methods and mempool mechanics
3. "Techniques for Modeling a Heavy-Tailed Bitcoin Fee Distribution" - Statistical approaches

## 12. Deliverables

1. **Code Repository**: Complete, documented codebase with models and data pipelines
2. **Trained Models**: Production-ready models for wait time and fee prediction
3. **Evaluation Report**: Comprehensive analysis of model performance
4. **Documentation**: User guides, API documentation, technical reports
5. **Presentation**: Final project presentation with results and insights

## 13. Risk Mitigation

### Technical Risks
- **Data Quality**: Implement robust validation and outlier detection
- **Model Overfitting**: Use proper cross-validation and regularization
- **Temporal Leakage**: Strict time-based train/test splits

### Data Risks
- **Missing Features**: Develop fallback models with available features
- **Data Availability**: Multiple data sources and collection methods

### Deployment Risks
- **Scalability**: Design modular, containerized architecture
- **Latency**: Optimize feature computation and model inference

## 14. Conclusion

This project addresses a critical problem in Bitcoin usability by developing accurate fee estimation models. Our two-stage approach, combined with impatience-based behavioral modeling, provides a novel and practical solution. The project combines rigorous academic methodology with practical engineering to deliver production-ready models that can improve the Bitcoin user experience.

---

**Repository**: https://github.com/mit-dci-cde-2025/final-project-proposals-kristian-s-team.git  
**Contact**: krisp@mit.edu

