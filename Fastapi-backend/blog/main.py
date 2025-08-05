
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt

from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel
from models import Fraud, Inspect,SecondHalf,FirstHalf1  # Import your Fraud and Inspect models
from schemas import FraudBase, InspectSchema  # Import your schemas
from database import SessionLocal, engine 
import logging
from models import ValidInfo, Base
from models import User1 
from database import SessionLocal
from sqlalchemy.exc import SQLAlchemyError
# Local imports
from database import engine, SessionLocal
import models
from fastapi.middleware.cors import CORSMiddleware
# Import your models and schemas
from models import Fraud, Inspect
from schemas import FraudBase, InspectSchema
from database import SessionLocal  # Assuming you have a database session factory
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import FraudList
from models import BillInfo1, LandInfo, ValidInfo, CustomerInfo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import schemas
from models import AAChart
from schemas import AAChartSchema
from schemas import ValidInfoSchema
from auth import get_password_hash, verify_password, create_access_token
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from typing import List
from database import SessionLocal, engine
from models import Base, CustomerInfo
from schemas import CustomerInfoSchema
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from collections import defaultdict
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import date
import models, schemas
from  database import get_db
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, kurtosis, skew, norm, zscore
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from models import LandInfo, CustomerInfo, ValidInfo, BillInfo1

import matplotlib
matplotlib.use('Agg')

 
# Initialize FastAPI app
 

# Create tables in database
models.Base.metadata.create_all(bind=engine)

# Password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT config
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
 
 

# ------------- Authentication Endpoints -------------
def hash_password(password):
    return pwd_context.hash(password)


class UserCreate(BaseModel):
    username: str
    password: str
    role: str

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = FastAPI()


@app.post("/register")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    try:
        # Check if user already exists
        existing_user = db.query(User1).filter(User1.username == user.username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already registered")

        # Hash the password before saving it
        hashed_password = hash_password(user.password)

        # Create a new User1 instance
        new_user = User1(username=user.username, hashed_password=hashed_password, role=user.role)

        # Add and commit the new user to the database
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return {"message": "User registered successfully", "user": new_user}

    except SQLAlchemyError as e:
        db.rollback()  # Ensure to rollback transaction on error
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
# ------------- Auth Helpers -------------
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"username": username, "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

 

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User1).filter(User1.username == form_data.username).first()
    if not user or not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer"}

def require_role(required_role: str):
    def role_dependency(user: dict = Depends(get_current_user)):
        if user["role"] != required_role:
            raise HTTPException(status_code=403, detail=f"Only {required_role} role can access this")
        return user
    return role_dependency
# ------------- Auth Helpers -------------
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"username": username, "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

 
@app.get("/Prepaid/aachart/{account_no}", response_model=Dict[str, Any])
def get_aachart_by_account(account_no: str, db: Session = Depends(get_db)):
    try:
        transactions = db.query(models.AAChart).filter(models.AAChart.accountno == account_no).order_by(models.AAChart.op_time).all()

        if not transactions:
            raise HTTPException(status_code=404, detail=f"No records found for account number {account_no}")

        # Process the transactions for response
        energy_values = []
        op_time_values = []
        total_amount_values = []
        total_montant_ttc_values = []
        total_amount = 0.0
        total_montant_ttc = 0.0
        tariff_counts = defaultdict(int)
        data = []

        for tx in transactions:
            op_time_str = tx.op_time.strftime('%Y-%m-%d') if tx.op_time else ""
            op_time_values.append(op_time_str)
            energy_values.append(tx.account_save_amount)

            amt = float(tx.total_amount or 0)
            ttc = float(tx.montant_ttc or 0)
            total_amount_values.append(amt)
            total_montant_ttc_values.append(ttc)
            total_amount += amt
            total_montant_ttc += ttc

            if tx.tariff:
                tariff_counts[tx.tariff] += 1

            data.append(AAChartSchema.model_validate(tx).model_dump())

        first = transactions[0]
        customer_info = {
            "CUST_NAME": first.cust_name,
            "ORDERSID": first.ordersid,
            "METERNO": first.meterno,
            "TOKEN": first.token,
            "OPERATOR": first.operator,
            "CODE": first.code,
            "POSID": first.posid,
            "OPERATOR_NAME": first.operator_name,
            "total_amount": total_amount,
            "total_montant_ttc": total_montant_ttc
        }

        return {
            "service_nbr": account_no,
            "bills": data,
            "summary": {
                "customer_info": customer_info,
                "energy_values": energy_values,
                "op_time_values": op_time_values,
                "total_amount_values": total_amount_values,
                "total_montant_ttc_values": total_montant_ttc_values,
                "tariff_labels": list(tariff_counts.keys()),
                "tariff_values": list(tariff_counts.values()),
                "timestamp_service_nbr_pairs": [{"timestamp": ts, "service_nbr": account_no} for ts in op_time_values]
            }
        }

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/Postpaid/contrat/{contrat}", response_model=Dict[str, Any])
def get_posters_by_contrat_admin(
    contrat: int,
    db: Session = Depends(get_db),
      # Can be removed if you want open access
):
    try:
        # No role check – any authenticated user can access this route
        posters = db.query(models.Poster).filter(models.Poster.Contrat == contrat).all()

        if not posters:
            raise HTTPException(status_code=404, detail=f"No records found for Contrat {contrat}")

        # Summary calculations
        total_consommation = sum(p.Consommation or 0 for p in posters)
        total_montant_ttc = sum(p.Montant_TTC or 0 for p in posters)
        total_paiement_caisse = sum(p.Paiement_Caisse or 0 for p in posters)
        remaining_amount = total_montant_ttc - total_paiement_caisse
        unique_ref_geo = list(set(p.Ref_Geo for p in posters if p.Ref_Geo))

        summary = {
            "Contrat": contrat,
            "Total_Consommation": total_consommation,
            "Total_Montant_TTC": total_montant_ttc,
            "Total_Paiement_Caisse": total_paiement_caisse,
            "Remaining_Amount": remaining_amount,
            "Unique_Ref_Geo": unique_ref_geo
        }

        return {
            "bills": [schemas.PosterSchema.model_validate(vars(p)) for p in posters],
            "summary": summary
        }

    except Exception as e:
        print("Error occurred:", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



@app.get("/inspectionndfraud/{service_no}", response_model=Dict[str, Any])
def inspectionndfraud_api(service_no: str, db: Session = Depends(get_db)):
    if not service_no:
        raise HTTPException(status_code=400, detail="No SERVICE_NO provided")

    try:
        # Fetch data from the 'Fraud' table
        fraud_data_2 = db.query(models.Fraud).filter(models.Fraud.SERVICE_NO == service_no).all()

        # If no data found in Fraud model, try fetching from Inspect model
        if not fraud_data_2:
            inspect_data = db.query(models.Inspect).filter(models.Inspect.SERVICE_NO == service_no).all()
            
            if not inspect_data:
                raise HTTPException(status_code=404, detail=f"No data found for SERVICE_NO: {service_no} in both Fraud and Inspect models")

            # Process inspect data and return it
            inspect_info = [schemas.InspectSchema.model_validate(inspect_entry).dict() for inspect_entry in inspect_data]
            return {
                'service_no': service_no,
                'inspect_data': inspect_info
            }

        # If fraud data is found, process and return it
        fraud_dates = []
        total_recovered_amounts = []
        power_values = []
        fraud_type_counts = defaultdict(int)
        appliance_code_counts = defaultdict(int)

        for entry in fraud_data_2:
            detection_date = entry.DETECTION_DATE

            if detection_date:
                if isinstance(detection_date, datetime):
                    detection_date = detection_date.date()
                fraud_dates.append(detection_date.strftime('%Y-%m-%d'))
            else:
                fraud_dates.append("N/A")

            total_recovered_amounts.append(entry.TOTAL_RECOVERED_AMOUNT or 0)
            power_values.append(entry.POWER or 0)

            fraud_type = entry.FRAUD_TYPE
            if fraud_type:
                fraud_type_counts[fraud_type] += 1

            appliance_code = entry.APPLIANCE_CODE
            if appliance_code:
                appliance_code_counts[appliance_code] += 1

        total_reference_count = len(fraud_data_2)
        total_power_sum = sum([entry.POWER for entry in fraud_data_2 if entry.POWER])

        return {
            'service_no': service_no,
            'fraud_data_2': [schemas.FraudBase.model_validate(entry).dict() for entry in fraud_data_2],
            'fraud_dates': fraud_dates,
            'total_recovered_amounts': total_recovered_amounts,
            'power_values': power_values,
            'fraud_type_counts': dict(fraud_type_counts),
            'appliance_code_counts': dict(appliance_code_counts),
            'total_reference_count': total_reference_count,
            'total_power_sum': total_power_sum
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.delete("/delete-user/{username}")
def delete_user(username: str, db: Session = Depends(get_db)):
    try:
        user = db.query(User1).filter(User1.username == username).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")

        db.delete(user)
        db.commit()

        return {"message": f"User '{username}' deleted successfully"}

    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error during deletion: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error during deletion: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/active-status-count")
def active_status_count(db: Session = Depends(get_db)):
    try:
        count = db.query(CustomerInfo).filter(
            CustomerInfo.service_status.in_(['ACTIVE (PENDING BILLING)', 'ACTIVE'])
        ).count()
        return JSONResponse(content={"label": "Active Status Count", "count": count}, status_code=200)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in active_status_count: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/inactive-status-count")
def inactive_status_count(db: Session = Depends(get_db)):
    try:
        count = db.query(CustomerInfo).filter(
            CustomerInfo.service_status.in_([
                'INACTIVE WITH BALANCE.', 'INACTIVE.',
                'INACTIVATION IN PROCESS.', 'IN PROCESS (PENDING CONNECTION)'
            ])
        ).count()
        return JSONResponse(content={"label": "Inactive Status Count", "count": count}, status_code=200)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in inactive_status_count: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

@app.get("/readwith-counts")
def readwith_counts(db: Session = Depends(get_db)):
    try:
        result = db.query(CustomerInfo.readwith, func.count(CustomerInfo.readwith))\
            .filter(CustomerInfo.readwith.in_(['SMARTPHONE', 'PREPAID', 'MMS']))\
            .group_by(CustomerInfo.readwith).all()
        data = [{"key": r[0], "value": r[1]} for r in result]
        return JSONResponse(content=data, status_code=200)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in readwith_counts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")



@app.get("/region-counts")
def region_counts(db: Session = Depends(get_db)):
    try:
        result = db.query(CustomerInfo.region, func.count(CustomerInfo.region))\
            .filter(CustomerInfo.region.in_([
                'DCUD', 'DRE', 'DRSANO', 'DCUY', 'DRSOM', 'DRONO', 'DRNEA', 'DRC', 'DRSM'
            ]))\
            .group_by(CustomerInfo.region).all()
        data = [{"key": r[0], "value": r[1]} for r in result]
        return JSONResponse(content=data, status_code=200)
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in region_counts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/tension-counts")
def tension_counts(db: Session = Depends(get_db)):
    try:
        tensions = [
            'LOW VOLTAGE( 220 )',
            'LOW VOLTAGE( 380 )',
            'MEDIUM VOLTAGE (15KV)',
            'MEDIUM VOLTAGE (30KV)',
            'MEDIUM VOLTAGE (10KV)'
        ]
        data = []
        for t in tensions:
            count = db.query(CustomerInfo).filter(CustomerInfo.tension == t).count()
            data.append({"key": t, "value": count})
        return JSONResponse(content=data, status_code=200)
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in tension_counts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

@app.get("/tariff-counts")
def tariff_counts(db: Session = Depends(get_db)):
    try:
        tariffs = [
            'LV', 'LV - DOMESTIC', 'LV - NON DOMESTIC', 'LV - ENEO  AGENT',
            'MV', 'MV - TARIF DE O-50 KW', 'LV - PUBLIC LIGHT',
            'MV - TARIF DE 55-500 KW', 'LV-CDE/CAMWATER/GLOBELEQ',
            'MV - TARIF DE 505 - 995 KW'
        ]
        result = db.query(CustomerInfo.tariff, func.count(CustomerInfo.tariff))\
            .filter(CustomerInfo.tariff.in_(tariffs))\
            .group_by(CustomerInfo.tariff).all()
        data = [{"key": r[0], "value": r[1]} for r in result]
        return JSONResponse(content=data, status_code=200)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in tariff_counts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")



def query_to_dataframe(query_result):
    data = [item.__dict__ for item in query_result]
    for d in data:
        d.pop('_sa_instance_state', None)
    return pd.DataFrame(data)


from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
@app.get("/all-data")
def get_all_data(db: Session = Depends(get_db)):
    try:
        # Query all tables
        bill_customers = db.query(BillInfo1).all()
        land_customers = db.query(LandInfo).all()
        validinfo_customers = db.query(ValidInfo).all()
        cust = db.query(CustomerInfo).all()

        # Convert to DataFrames
        df_bill = query_to_dataframe(bill_customers)
        df_land = query_to_dataframe(land_customers)
        df_validinfo = query_to_dataframe(validinfo_customers)
        df_customer = query_to_dataframe(cust)

        # Counts and aggregates
        total_services = int(len(df_land))
        fraud_cases = int(df_land[df_land['Found_fraud'] > 0].shape[0])
        fraud_total_payment = float(df_land[df_land['Found_fraud'] > 0]['Total_Payment'].sum())
        avg_consumption = float(df_land['Avg_Consumption'].mean() or 0)

        # Tariff distribution
        tariff_counts_df = df_land['TARIFF'].value_counts().reset_index(name='count')
        tariff_counts_df.rename(columns={'TARIFF': 'tariff'}, inplace=True)
        tariff_labels = tariff_counts_df['tariff'].tolist()
        tariff_counts = tariff_counts_df['count'].tolist()

        # Payment mode distribution
        payment_counts_df = df_land['payment_mode'].value_counts().reset_index(name='count')
        payment_mode_labels = payment_counts_df['payment_mode'].tolist()
        payment_mode_counts = payment_counts_df['count'].tolist()

        # Prepaid vs Postpaid fraud
        prepaid_fraud_count = int(df_land[(df_land['payment_mode'] == 'PREPAID') & (df_land['Found_fraud'] > 0)].shape[0])
        postpaid_fraud_count = int(df_land[(df_land['payment_mode'] == 'POSTPAID') & (df_land['Found_fraud'] > 0)].shape[0])

        # Phase-wise fraud
        phase_counts_df = df_land[df_land['Found_fraud'] > 0]['PHASE'].value_counts().reset_index(name='count')
        phase_counts_df.rename(columns={'PHASE': 'phase_label'}, inplace=True)
        phase_labels = phase_counts_df['phase_label'].tolist()
        phase_counts = phase_counts_df['count'].tolist()

        # Tension-wise fraud
        tension_counts_df = df_land[df_land['Found_fraud'] > 0]['TENSION'].value_counts().reset_index(name='count')
        tension_labels = tension_counts_df['TENSION'].tolist()
        tension_counts = tension_counts_df['count'].tolist()

        # Readwith, power subscribed, service status
        readwith_counts_df = df_customer['readwith'].value_counts().reset_index(name='count')
        readwith_labels = readwith_counts_df['readwith'].tolist()
        readwith_counts = readwith_counts_df['count'].tolist()

        power_subscribed_counts_df = df_customer['power_subscribed'].value_counts().reset_index(name='count')
        power_subscribed_labels = power_subscribed_counts_df['power_subscribed'].tolist()
        power_subscribed_counts = power_subscribed_counts_df['count'].tolist()

        service_status_counts_df = df_customer['service_status'].value_counts().reset_index(name='count')
        service_status_labels = service_status_counts_df['service_status'].tolist()
        service_status_counts = service_status_counts_df['count'].tolist()

        # Heatmap image
        correlation_matrix = df_validinfo.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        heatmap_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # ANOVA
        df = df_bill
        exclude_columns = ['CUSTOMER_NAME', 'METER_NBR', 'REF_GEO', 'ENROLMENT_DATE', 'CANCEL_DATE', 'Date_de_Facturation', 'SERVICE_NBR']
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        columns_for_anova = [col for col in numeric_columns if col not in exclude_columns]

        anova_scores = []
        for column in columns_for_anova:
            try:
                if df[column].var() == 0:
                    continue
                temp_df = pd.DataFrame({
                    'Has_Fraud': df['Has_Fraud'].astype(int),
                    'variable': df[column].astype(float)
                }).dropna()
                if len(temp_df) > 0:
                    fraud_group = temp_df[temp_df['Has_Fraud'] == 1]['variable']
                    non_fraud_group = temp_df[temp_df['Has_Fraud'] == 0]['variable']
                    if len(fraud_group) > 0 and len(non_fraud_group) > 0:
                        f_stat, _ = f_oneway(fraud_group, non_fraud_group)
                        anova_scores.append({'Specs': column, 'F_Value': float(f_stat)})
            except Exception as e:
                print(f"ANOVA error on {column}: {e}")
                continue

        anova_df = pd.DataFrame(anova_scores).sort_values(by='F_Value', ascending=False).head(20) if anova_scores else pd.DataFrame(columns=['Specs', 'F_Value'])
        anova_specs = anova_df['Specs'].tolist()
        anova_fvalues = anova_df['F_Value'].tolist()

        # Skew & Kurtosis
        numeric_df = df_validinfo.select_dtypes(include=np.number).dropna(axis=1, how='all')

        def interpret_skew(value):
            if value > 1:
                return "Highly right-skewed"
            elif value > 0.5:
                return "Moderately right-skewed"
            elif value < -1:
                return "Highly left-skewed"
            elif value < -0.5:
                return "Moderately left-skewed"
            else:
                return "Approximately symmetric"

        def interpret_kurtosis(value):
            if value > 3:
                return "Leptokurtic (Heavy tails, more outliers)"
            elif value < -1:
                return "Platykurtic (Light tails, fewer outliers)"
            elif value < 1:
                return "Mesokurtic (Normal-like tails)"
            else:
                return "Slightly Leptokurtic"

        skew_kurtosis_results = []
        for column in numeric_df.columns:
            try:
                col_data = numeric_df[column].dropna()
                if len(col_data) > 1 and col_data.var() > 0:
                    s = skew(col_data, nan_policy='omit')
                    k = kurtosis(col_data, fisher=True, nan_policy='omit')
                    skew_kurtosis_results.append({
                        "column": column,
                        "skewness": round(s, 4),
                        "skewness_status": interpret_skew(s),
                        "kurtosis": round(k, 4),
                        "kurtosis_status": interpret_kurtosis(k)
                    })
            except Exception as e:
                print(f"Skew/Kurtosis error on {column}: {e}")
                continue

        # Confidence Intervals
        df3 = numeric_df
        numerical_cols = df3.select_dtypes(include=np.number).columns
        confidence = 0.95
        z = norm.ppf(1 - (1 - confidence) / 2)
        n = len(df3)

        ci_results = []
        for col in numerical_cols:
            mean = df3[col].mean()
            std = df3[col].std(ddof=1)
            margin = z * (std / np.sqrt(n))
            lower = mean - margin
            upper = mean + margin
            ci_results.append({
                'Feature': col,
                'Mean': round(mean, 4),
                'CI_Lower': round(lower, 4),
                'CI_Upper': round(upper, 4)
            })

        ci_df = pd.DataFrame(ci_results)
        confidence_intervals = ci_df.to_dict(orient='records')

        # Z-score & Outliers
        if not df3.empty and not numerical_cols.empty:
            z_scores_df = df3[numerical_cols.tolist()].apply(zscore)
            outliers = (np.abs(z_scores_df) > 3)
            outlier_counts = outliers.sum().reset_index()
            outlier_counts.columns = ['Feature', 'Outlier_Count']
            outlier_data = outlier_counts.to_dict(orient='records')
        else:
            outlier_data = []

        # Linear Regression
        try:
            features = [
                'PHASE', 'TARIFF', 'POWER_SUSCRIBED', 'TENSION', 'DIVISION',
                'ACTIVITY_CMS', 'REGION', 'AGENCY', 'payment_mode', 'IsActive',
                'Tenure_Days', 'Avg_Consumption', 'Max_Consumption', 'Total_Payment',
                'Spike_Count', 'Has_inspected', 'inspection_count',
                'fraud_found_category', 'SERVICE_NBR'
            ]
            target = 'Found_fraud'
            df_model = df3[features + [target]].dropna()
            X = df_model[features]
            y = df_model[target]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LinearRegression()
            model.fit(X_scaled, y)
            coefficients = model.coef_
            intercept = model.intercept_
            regression_results = pd.DataFrame({
                'Feature': features,
                'Coefficient': coefficients
            }).sort_values(by='Coefficient', ascending=False)

            linear_regression_data = {
                'intercept': round(intercept, 4),
                'coefficients': regression_results.to_dict(orient='records')
            }
        except Exception as e:
            print(f"Linear regression error: {e}")
            linear_regression_data = {
                'intercept': None,
                'coefficients': []
            }

        # Final response
        response_data = {
            "total_services": total_services,
            "fraud_cases": fraud_cases,
            "fraud_total_payment": fraud_total_payment,
            "avg_consumption": avg_consumption,
            "tariff_labels": tariff_labels,
            "tariff_counts": tariff_counts,
            "payment_mode_labels": payment_mode_labels,
            "payment_mode_counts": payment_mode_counts,
            "prepaid_fraud_count": prepaid_fraud_count,
            "postpaid_fraud_count": postpaid_fraud_count,
            "phase_labels": phase_labels,
            "phase_counts": phase_counts,
            "tension_labels": tension_labels,
            "tension_counts": tension_counts,
            'readwith_chart_data': {'labels': readwith_labels, 'data': readwith_counts},
            'power_subscribed_chart_data': {'labels': power_subscribed_labels, 'data': power_subscribed_counts},
            'service_status_chart_data': {'labels': service_status_labels, 'data': service_status_counts},
            "heatmap_image": heatmap_image,
            'anova_specs': anova_specs,
            'anova_fvalues': anova_fvalues,
            'skew_kurtosis_results': skew_kurtosis_results,
            'confidence_intervals': confidence_intervals,
            'outlier_data': outlier_data,
            'linear_regression_data': linear_regression_data,
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the request: {str(e)}"
        )
 

class FraudRequest(BaseModel):
    service_nbr: int
@app.get("/predict_fraud_db/{service_nbr}")
def predict_fraud_by_service_nbr(service_nbr: int, db: Session = Depends(get_db)):
    # Query both tables
    df1_query = db.query(FirstHalf1).all()
    df2_query = db.query(SecondHalf).all()

    df1 = query_to_dataframe(df1_query)
    df2 = query_to_dataframe(df2_query)

    if df1.empty or df2.empty:
        raise HTTPException(status_code=500, detail="Data tables are empty.")

    # Convert SERVICE_NBR to string for matching
    df2['SERVICE_NBR'] = df2['SERVICE_NBR'].astype(str)
    service_nbr_to_search = str(service_nbr)
    matching_row = df2.loc[df2['SERVICE_NBR'] == service_nbr_to_search]

    if matching_row.empty:
        raise HTTPException(status_code=404, detail=f"SERVICE_NBR {service_nbr} not found.")

    # Prepare training data
    X = df1.drop(['Found_fraud', 'SERVICE_NBR'], axis=1, errors='ignore')
    y = df1['Found_fraud']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Prepare prediction features
    prediction_features = matching_row.drop(['Found_fraud', 'SERVICE_NBR'], axis=1, errors='ignore')
    prediction_features = prediction_features[X.columns]  # align columns

    # Make prediction
    prediction = model.predict(prediction_features)

    result = "Fraud predicted" if prediction[0] == 1 else "No fraud predicted"

    # If fraud is predicted, insert into the fraudlist table
    if prediction[0] == 1:
        try:
            # Insert into fraudlist table
            new_fraud_entry = FraudList(service_nbr=service_nbr)
            db.add(new_fraud_entry)
            db.commit()  # Commit the transaction
        except Exception as e:
            db.rollback()  # Rollback in case of error
            raise HTTPException(status_code=500, detail=f"Error inserting into fraudlist: {str(e)}")

    return {
        "service_nbr": service_nbr,
        "prediction": result
    }


@app.get("/fraudlist", response_model=List[schemas.FraudListSchema])
def get_all_fraudlist(db: Session = Depends(get_db)):
    try:
        fraudlist_records = db.query(FraudList).all()

        if not fraudlist_records:
            raise HTTPException(status_code=404, detail="No fraudlist records found.")

        return fraudlist_records

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
# Add CORS middleware


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from fastapi import Query

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from io import BytesIO
import base64
@app.post("/predict_fraud_csv/")
async def predict_fraud_from_csv(
    file: UploadFile = File(...),
    region: str | None = Query(None, description="Filter by REGION before prediction")
):
    try:
        df = pd.read_csv(BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    # Normalize columns to lowercase
    df.columns = df.columns.str.lower()

    # Check and filter by region if provided
    if region:
        if 'region' not in df.columns:
            raise HTTPException(status_code=400, detail="'region' column not found in CSV.")
        # Filter rows by region (case-insensitive)
        df = df[df['region'].str.lower() == region.lower()]
        if df.empty:
            raise HTTPException(status_code=400, detail=f"No data found for region '{region}'.")

    # Determine target column name
    if 'found_fraud' in df.columns:
        target_col = 'found_fraud'
    elif 'fraud_found_category' in df.columns:
        target_col = 'fraud_found_category'
    else:
        target_col = None

    # Drop rows with null target if present
    if target_col:
        df = df.dropna(subset=[target_col])
        if df.empty:
            raise HTTPException(status_code=400, detail=f"No rows left after dropping null '{target_col}'.")

    # Columns to label encode
    categorical_cols = ['phase', 'tariff', 'tension', 'division', 'activity_cms',
                        'region', 'agency', 'payment_mode', 'isactive']

    for col in categorical_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    drop_cols = ['service_nbr']
    if target_col:
        drop_cols.append(target_col)

    cols_to_drop = [col for col in drop_cols if col in df.columns]
    features = df.drop(columns=cols_to_drop)

    if features.empty:
        raise HTTPException(status_code=400, detail="No features left for prediction after dropping irrelevant columns.")

    scaled_features = StandardScaler().fit_transform(features)
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)

    metrics = {}

    if target_col:
        X = df_scaled
        y = df[target_col].astype(int)

        # Train-test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        predictions = model.predict(X)

        acc = accuracy_score(y, predictions)
        cm = confusion_matrix(y, predictions).tolist()
        report = classification_report(y, predictions, output_dict=True)

        metrics = {
            "accuracy": round(acc, 4),
            "confusion_matrix": cm,
            "classification_report": report
        }

    else:
        # No target → dummy predictions
        dummy_labels = [0] * len(df_scaled)
        model = LogisticRegression(max_iter=1000)
        model.fit(df_scaled, dummy_labels)
        predictions = model.predict(df_scaled)

        metrics = {
            "note": "No 'found_fraud' or 'fraud_found_category' column provided. Dummy model used for predictions.",
            "accuracy": None,
            "confusion_matrix": None,
            "classification_report": None
        }

    df['predicted_fraud'] = predictions

    # Encode the output CSV back to base64
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    encoded_csv = base64.b64encode(output.read()).decode('utf-8')

    return JSONResponse(content={
        "region_filtered": region if region else "all",
        "num_rows": len(df),
        "metrics": metrics,
        "file": encoded_csv,
        "filename": f"fraud_predictions_{region if region else 'all'}.csv"
    })
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)