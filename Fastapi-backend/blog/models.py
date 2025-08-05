from sqlalchemy import Column, BigInteger, Float, String, Date
from database import Base

class Poster(Base):
    __tablename__ = "poster"

    Numero_Facture = Column(BigInteger, primary_key=True, index=True)
    Contrat = Column(BigInteger, nullable=True)
    Consommation = Column(Float, nullable=True)
    Montant_TTC = Column(Float, nullable=True)
    Paiement_Caisse = Column(Float, nullable=True)
    Type_Facture = Column(String(255), nullable=True)
    Ref_Geo = Column(String(255), nullable=True)
    Date_de_Facturation = Column(Date, nullable=True)
    Type_de_Facturation = Column(String(255), nullable=True)


from sqlalchemy import Column, Integer, String, Text
from database import Base

class Blog(Base):
    __tablename__ = 'blogs'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    body = Column(Text, nullable=False)


from sqlalchemy import Column, BigInteger, String, Double, DateTime
from database import Base

class AAChart(Base):
    __tablename__ = "aachart"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tariff = Column(String(255), nullable=True)
    cust_name = Column(String(255), nullable=True)
    ordersid = Column(String(255), nullable=True)
    meterno = Column(String(255), nullable=True)
    accountno = Column(String(255), unique=True, nullable=False)
    total_amount = Column(Double, nullable=True)
    montant_ttc = Column(Double, nullable=True)
    montant_client_ht = Column(Double, nullable=True)
    montant_client_tva = Column(Double, nullable=True)
    energy = Column(Double, nullable=True)
    token = Column(String(255), nullable=True)
    account_pay_amount = Column(Double, nullable=True)
    account_save_amount = Column(Double, nullable=True)
    op_time = Column(DateTime, nullable=True)
    montant_entreprise_tva = Column(Double, nullable=True)
    montant_entreprise_hors_taxe = Column(Double, nullable=True)
    operator = Column(String(255), nullable=True)
    code = Column(String(255), nullable=True)
    posid = Column(String(255), nullable=True)
    operator_name = Column(String(255), nullable=True)
    class Config:
        orm_mode = True


from sqlalchemy import Column, Integer, String, Float, Date, Text
from database import Base

class Fraud(Base):
    __tablename__ = "fraud"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    SERVICE_NO = Column(String(20), nullable=True)
    FRAUD_NUMBER = Column(Float, nullable=True)
    DETECTION_DATE = Column(Date, nullable=True)
    FRAUD_TYPE = Column(String(255), nullable=True)
    OBSERVATIONS = Column(Text, nullable=True)
    REFERENCE_DIRECTION = Column(String(255), nullable=True)
    TOTAL_RECOVERED_AMOUNT = Column(Float, nullable=True)
    RECORD_NUMBER = Column(String(50), nullable=True)
    STATUS = Column(String(100), nullable=True)
    APPLIANCE_CODE = Column(String(50), nullable=True)
    POWER = Column(Float, nullable=True)
    QUANTITY = Column(Float, nullable=True)
    HOURS_OF_USE_PER_DAY = Column(String(50), nullable=True)

    class Config:
        from_attributes = True  # Enables from_attributes functionality
        orm_mode = True  # Still necessary for ORM compatibility
from sqlalchemy import Column, Integer, String, Float
from database import Base

class Inspect(Base):
    __tablename__ = "inspect"

    id = Column(Integer, primary_key=True, index=True)
    SERVICE_NO = Column(String(20), nullable=True)
    METER_NUMBER = Column(String(50), nullable=True)
    SUBSCRIPTION_LOAD = Column(Float, nullable=True)
    NB_ANOMALIES = Column(Integer, nullable=True)
    NB_ALREADY_CONTROLLED = Column(Integer, nullable=True)
    ALREADY_DETECTED_FRAUD = Column(String(255), nullable=True)
    class Config:
        from_attributes = True  # Enables from_attributes functionality
        orm_mode = True 

from sqlalchemy import Column, Integer, String
from database import Base

class User1(Base):
    __tablename__ = 'users1'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)  # Ensure 'unique' is set here
    hashed_password = Column(String(255), nullable=False)
    role = Column(String, default="technician")

from sqlalchemy import Column, Integer, String, Float, Date, BigInteger
from database import Base

class CustomerInfo(Base):
    __tablename__ = "customer_info"

    region = Column(String(100), nullable=True)
    division = Column(String(100), nullable=True)
    agency = Column(String(100), nullable=True)
    customer_name = Column(String(200), nullable=True)
    service_nbr = Column(BigInteger, primary_key=True, index=True)
    meter_nbr = Column(String(50), nullable=True)
    install_date = Column(Date, nullable=True)
    ref_geo = Column(String(100), nullable=True)
    itinerary = Column(Float, nullable=True)
    service_status = Column(String(50), nullable=True)
    enrolment_date = Column(Date, nullable=True)
    cancel_date = Column(Date, nullable=True)
    power_subscribed = Column(Float, nullable=True)
    tension = Column(String(50), nullable=True)
    phase = Column(String(50), nullable=True)
    tariff = Column(String(50), nullable=True)
    activity_cms = Column(String(100), nullable=True)
    readwith = Column(String(50), nullable=True)
    segment = Column(String(100), nullable=True)
    value_koppen = Column(Integer, nullable=True)
    desc_koppen = Column(String(100), nullable=True)
    code_koppen = Column(String(50), nullable=True)


from sqlalchemy import Column, Integer, Float, BigInteger
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ValidInfo(Base):
    __tablename__ = 'valid_info'
    
    PHASE = Column(Integer, nullable=True)
    TARIFF = Column(Integer, nullable=True)
    POWER_SUSCRIBED = Column(Float, nullable=True)
    TENSION = Column(Integer, nullable=True)
    DIVISION = Column(Integer, nullable=True)
    ACTIVITY_CMS = Column(Integer, nullable=True)
    REGION = Column(Integer, nullable=True)
    AGENCY = Column(Integer, nullable=True)
    payment_mode = Column(Integer, nullable=True)
    IsActive = Column(Integer, nullable=True)
    Tenure_Days = Column(Float, nullable=True)
    Avg_Consumption = Column(Float, nullable=True)
    Max_Consumption = Column(Float, nullable=True)
    Total_Payment = Column(Float, nullable=True)
    Spike_Count = Column(Float, nullable=True)
    Has_inspected = Column(Integer, nullable=True)
    inspection_count = Column(Integer, nullable=True)
    fraud_found_category = Column(Integer, nullable=True)
    Found_fraud = Column(Integer, nullable=True)
    SERVICE_NBR = Column(BigInteger, primary_key=True, nullable=False)

    class Config:
        orm_mode = True
        from_attributes = True
    




from sqlalchemy import Column, Integer, BigInteger, String, Float
from database import Base  # make sure Base is from declarative_base()

class LandInfo(Base):
    __tablename__ = "land_info"

    SERVICE_NBR = Column(BigInteger, primary_key=True, index=True)
    PHASE = Column(String(50), nullable=True)
    TARIFF = Column(String(100), nullable=True)
    POWER_SUSCRIBED = Column(Float, nullable=True)
    TENSION = Column(String(50), nullable=True)
    DIVISION = Column(String(100), nullable=True)
    ACTIVITY_CMS = Column(String(100), nullable=True)
    REGION = Column(String(100), nullable=True)
    AGENCY = Column(String(100), nullable=True)
    payment_mode = Column(String(100), nullable=True)
    IsActive = Column(Integer, nullable=True)
    Tenure_Days = Column(Integer, nullable=True)
    Avg_Consumption = Column(Float, nullable=True)
    Max_Consumption = Column(Integer, nullable=True)
    Total_Payment = Column(Integer, nullable=True)
    Spike_Count = Column(Integer, nullable=True)
    Has_inspected = Column(Integer, nullable=True)
    inspection_count = Column(Integer, nullable=True)
    fraud_found_category = Column(Integer, nullable=True)
    Found_fraud = Column(Integer, nullable=True)

    class Config:
        orm_mode = True
        from_attributes = True


from sqlalchemy import Column, Integer, String, BigInteger, Float, Boolean, Date
from database import Base  # Assuming you use `Base = declarative_base()`

class BillInfo1(Base):
    __tablename__ = 'bill_info'

    id = Column(Integer, primary_key=True, index=True)
    REGION = Column(String(100), nullable=True)
    DIVISION = Column(String(100), nullable=True)
    AGENCY = Column(String(100), nullable=True)
    CUSTOMER_NAME = Column(String(200), nullable=True)
    SERVICE_NBR = Column(BigInteger, index=True, nullable=True)
    METER_NBR = Column(String(50), nullable=True)
    INSTALL_DATE = Column(Date, nullable=True)
    REF_GEO = Column(String(100), nullable=True)
    ITINERARY = Column(Float, nullable=True)
    SERVICE_STATUS = Column(String(50), nullable=True)
    ENROLMENT_DATE = Column(Date, nullable=True)
    CANCEL_DATE = Column(Date, nullable=True)
    POWER_SUSCRIBED = Column(Float, nullable=True)
    TENSION = Column(String(50), nullable=True)
    PHASE = Column(String(50), nullable=True)
    TARIFF = Column(String(50), nullable=True)
    ACTIVITY_CMS = Column(String(100), nullable=True)
    READWITH = Column(String(50), nullable=True)
    SEGMENT = Column(String(100), nullable=True)
    valueKoppen = Column(Integer, nullable=True)
    descKoppen = Column(String(100), nullable=True)
    codeKoppen = Column(String(50), nullable=True)
    Tenure_Days = Column(Float, nullable=True)
    IsActive = Column(Boolean, nullable=True)
    Power_Category = Column(Integer, nullable=True)
    Numero_Facture = Column(BigInteger, nullable=True)
    Contrat = Column(BigInteger, nullable=True)
    Consommation = Column(Float, nullable=True)
    Montant_TTC = Column(Float, nullable=True)
    Paiement_Caisse = Column(Float, nullable=True)
    Type_Facture = Column(String(100), nullable=True)
    Date_de_Facturation = Column(Date, nullable=True)
    Type_de_Facturation = Column(String(50), nullable=True)
    unique_fraud_count = Column(Integer, nullable=True)
    Has_Fraud = Column(Boolean, nullable=True)
    Has_Inspection = Column(Boolean, nullable=True)

    class Config:
        orm_mode = True
        from_attributes = True


from sqlalchemy import Column, String, Integer, Boolean, DECIMAL
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class SecondHalf(Base):
    __tablename__ = '2ndhalf'

    SERVICE_NBR = Column(String(50), primary_key=True, index=True)
    PHASE = Column(String(20))
    TARIFF = Column(String(50))
    POWER_SUSCRIBED = Column(String(20))
    TENSION = Column(String(20))
    DIVISION = Column(String(100))
    ACTIVITY_CMS = Column(String(100))
    REGION = Column(String(50))
    AGENCY = Column(String(100))
    payment_mode = Column(String(50))
    IsActive = Column(Boolean)
    Tenure_Days = Column(Integer)
    Avg_Consumption = Column(DECIMAL(12, 2))
    Max_Consumption = Column(DECIMAL(12, 2))
    Total_Payment = Column(DECIMAL(14, 2))
    Spike_Count = Column(Integer)
    Has_inspected = Column(Boolean)
    inspection_count = Column(Integer)
    fraud_found_category = Column(String(100))
    Found_fraud = Column(Boolean)


class FirstHalf1(Base):
    __tablename__ = '11sthalf'

    SERVICE_NBR = Column(String(50), primary_key=True, index=True)
    PHASE = Column(String(20))
    TARIFF = Column(String(50))
    POWER_SUSCRIBED = Column(String(20))
    TENSION = Column(String(20))
    DIVISION = Column(String(100))
    ACTIVITY_CMS = Column(String(100))
    REGION = Column(String(50))
    AGENCY = Column(String(100))
    payment_mode = Column(String(50))
    IsActive = Column(Boolean)
    Tenure_Days = Column(Integer)
    Avg_Consumption = Column(DECIMAL(12, 2))
    Max_Consumption = Column(DECIMAL(12, 2))
    Total_Payment = Column(DECIMAL(14, 2))
    Spike_Count = Column(Integer)
    Has_inspected = Column(Boolean)
    inspection_count = Column(Integer)
    fraud_found_category = Column(String(100))
    Found_fraud = Column(Boolean)

class FraudList(Base):
    __tablename__ = 'fraudlist'
    service_nbr = Column(Integer, primary_key=True)


class History(Base):
    __tablename__ = 'History'
    service_nbr = Column(String(255), primary_key=True)
from sqlalchemy import Column, String, Integer, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
class Model(Base):
    __tablename__ = "models"
    __table_args__ = {"schema": "cluster"}  # Specify the schema

    PHASE = Column(String(50))
    TARIFF = Column(String(50))
    POWER_SUSCRIBED = Column(Float)
    TENSION = Column(String(50))
    DIVISION = Column(String(100))
    ACTIVITY_CMS = Column(String(100), nullable=True)
    REGION = Column(String(100))
    AGENCY = Column(String(100))
    payment_mode = Column(String(50))
    IsActive = Column(Integer)
    Tenure_Days = Column(Integer)
    Avg_Consumption = Column(Float)
    Max_Consumption = Column(Float)
    Total_Payment = Column(Float)
    Spike_Count = Column(Integer)
    Has_inspected = Column(Integer)
    inspection_count = Column(Integer)
    fraud_found_category = Column(Integer)
    Found_fraud = Column(Integer)
    SERVICE_NBR = Column(BigInteger, primary_key=True)