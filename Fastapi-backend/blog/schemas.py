from pydantic import BaseModel
from datetime import date
from typing import Optional

class PosterSchema(BaseModel):
    Numero_Facture: int
    Contrat: Optional[int] = None
    Consommation: Optional[float] = None
    Montant_TTC: Optional[float] = None
    Paiement_Caisse: Optional[float] = None
    Type_Facture: Optional[str] = None
    Ref_Geo: Optional[str] = None
    Date_de_Facturation: Optional[date] = None
    Type_de_Facturation: Optional[str] = None

    class Config:
        orm_mode = True


from pydantic import BaseModel
class BlogSchema(BaseModel):
    title: str
    body: str

    class Config:
        orm_mode = True

from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class AAChartSchema(BaseModel):
    id: int
    tariff: Optional[str]
    cust_name: Optional[str]
    ordersid: Optional[str]
    meterno: Optional[str]
    accountno: str
    total_amount: Optional[float]
    montant_ttc: Optional[float]
    montant_client_ht: Optional[float]
    montant_client_tva: Optional[float]
    energy: Optional[float]
    token: Optional[str]
    account_pay_amount: Optional[float]
    account_save_amount: Optional[float]
    op_time: Optional[datetime]
    montant_entreprise_tva: Optional[float]
    montant_entreprise_hors_taxe: Optional[float]
    operator: Optional[str]
    code: Optional[str]
    posid: Optional[str]
    operator_name: Optional[str]

    model_config = {
        "from_attributes": True
    }

from pydantic import BaseModel
from typing import Optional
from datetime import date

class FraudBase(BaseModel):
    SERVICE_NO: Optional[str]
    FRAUD_NUMBER: Optional[float]
    DETECTION_DATE: Optional[date]
    FRAUD_TYPE: Optional[str]
    OBSERVATIONS: Optional[str]
    REFERENCE_DIRECTION: Optional[str]
    TOTAL_RECOVERED_AMOUNT: Optional[float]
    RECORD_NUMBER: Optional[str]
    STATUS: Optional[str]
    APPLIANCE_CODE: Optional[str]
    POWER: Optional[float]
    QUANTITY: Optional[float]
    HOURS_OF_USE_PER_DAY: Optional[str]


     
    class Config:
        from_attributes = True  # Enables from_attributes functionality
        orm_mode = True  # Ensures ORM compatibility
class FraudCreateSchema(FraudBase):
    pass  # Use all fields from FraudBase

class FraudSchema(FraudBase):
    id: int

    class Config:
        orm_mode = True


from pydantic import BaseModel
from typing import Optional

class InspectSchema(BaseModel):
    id: int
    SERVICE_NO: Optional[str] = None
    METER_NUMBER: Optional[str] = None
    SUBSCRIPTION_LOAD: Optional[float] = None
    NB_ANOMALIES: Optional[int] = None
    NB_ALREADY_CONTROLLED: Optional[int] = None
    ALREADY_DETECTED_FRAUD: Optional[str] = None

    class Config:
        orm_mode = True
        from_attributes = True

 

class UserCreate(BaseModel):
    username: str
    password: str
    role: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    username: str
    role: str

    class Config:
        orm_mode = True



from pydantic import BaseModel
from typing import Optional
from datetime import date

class CustomerInfoSchema(BaseModel):
    region: Optional[str]
    division: Optional[str]
    agency: Optional[str]
    customer_name: Optional[str]
    service_nbr: int
    meter_nbr: Optional[str]
    install_date: Optional[date]
    ref_geo: Optional[str]
    itinerary: Optional[float]
    service_status: Optional[str]
    enrolment_date: Optional[date]
    cancel_date: Optional[date]
    power_subscribed: Optional[float]
    tension: Optional[str]
    phase: Optional[str]
    tariff: Optional[str]
    activity_cms: Optional[str]
    readwith: Optional[str]
    segment: Optional[str]
    value_koppen: Optional[int]
    desc_koppen: Optional[str]
    code_koppen: Optional[str]

    class Config:
        orm_mode = True

from pydantic import BaseModel
from typing import Optional

class ValidInfoSchema(BaseModel):
    PHASE: Optional[int]
    TARIFF: Optional[int]
    POWER_SUSCRIBED: Optional[float]
    TENSION: Optional[int]
    DIVISION: Optional[int]
    ACTIVITY_CMS: Optional[int]
    REGION: Optional[int]
    AGENCY: Optional[int]
    payment_mode: Optional[int]
    IsActive: Optional[int]
    Tenure_Days: Optional[float]
    Avg_Consumption: Optional[float]
    Max_Consumption: Optional[float]
    Total_Payment: Optional[float]
    Spike_Count: Optional[float]
    Has_inspected: Optional[int]
    inspection_count: Optional[int]
    fraud_found_category: Optional[int]
    Found_fraud: Optional[int]
    SERVICE_NBR: int

    class Config:
        orm_mode = True



class LandInfoSchema(BaseModel):
    SERVICE_NBR: int
    PHASE: Optional[str] = None
    TARIFF: Optional[str] = None
    POWER_SUSCRIBED: Optional[float] = None
    TENSION: Optional[str] = None
    DIVISION: Optional[str] = None
    ACTIVITY_CMS: Optional[str] = None
    REGION: Optional[str] = None
    AGENCY: Optional[str] = None
    payment_mode: Optional[str] = None
    IsActive: Optional[int] = None
    Tenure_Days: Optional[int] = None
    Avg_Consumption: Optional[float] = None
    Max_Consumption: Optional[int] = None
    Total_Payment: Optional[int] = None
    Spike_Count: Optional[int] = None
    Has_inspected: Optional[int] = None
    inspection_count: Optional[int] = None
    fraud_found_category: Optional[int] = None
    Found_fraud: Optional[int] = None

    class Config:
        orm_mode = True
        from_attributes = True

from pydantic import BaseModel
from typing import Optional
from datetime import date

class BillInfo1Base(BaseModel):
    REGION: Optional[str]
    DIVISION: Optional[str]
    AGENCY: Optional[str]
    CUSTOMER_NAME: Optional[str]
    SERVICE_NBR: Optional[int]
    METER_NBR: Optional[str]
    INSTALL_DATE: Optional[date]
    REF_GEO: Optional[str]
    ITINERARY: Optional[float]
    SERVICE_STATUS: Optional[str]
    ENROLMENT_DATE: Optional[date]
    CANCEL_DATE: Optional[date]
    POWER_SUSCRIBED: Optional[float]
    TENSION: Optional[str]
    PHASE: Optional[str]
    TARIFF: Optional[str]
    ACTIVITY_CMS: Optional[str]
    READWITH: Optional[str]
    SEGMENT: Optional[str]
    valueKoppen: Optional[int]
    descKoppen: Optional[str]
    codeKoppen: Optional[str]
    Tenure_Days: Optional[float]
    IsActive: Optional[bool]
    Power_Category: Optional[int]
    Numero_Facture: Optional[int]
    Contrat: Optional[int]
    Consommation: Optional[float]
    Montant_TTC: Optional[float]
    Paiement_Caisse: Optional[float]
    Type_Facture: Optional[str]
    Date_de_Facturation: Optional[date]
    Type_de_Facturation: Optional[str]
    unique_fraud_count: Optional[int]
    Has_Fraud: Optional[bool]
    Has_Inspection: Optional[bool]

class BillInfo1Create(BillInfo1Base):
    pass

class BillInfo1Out(BillInfo1Base):
    id: int

    class Config:
        orm_mode = True


from pydantic import BaseModel
from typing import Optional

class SecondHalfBase(BaseModel):
    PHASE: Optional[str]
    TARIFF: Optional[str]
    POWER_SUSCRIBED: Optional[str]
    TENSION: Optional[str]
    DIVISION: Optional[str]
    ACTIVITY_CMS: Optional[str]
    REGION: Optional[str]
    AGENCY: Optional[str]
    payment_mode: Optional[str]
    IsActive: Optional[bool]
    Tenure_Days: Optional[int]
    Avg_Consumption: Optional[float]
    Max_Consumption: Optional[float]
    Total_Payment: Optional[float]
    Spike_Count: Optional[int]
    Has_inspected: Optional[bool]
    inspection_count: Optional[int]
    fraud_found_category: Optional[str]
    Found_fraud: Optional[bool]

class SecondHalfCreate(SecondHalfBase):
    SERVICE_NBR: str

class SecondHalfResponse(SecondHalfBase):
    SERVICE_NBR: str

    class Config:
        orm_mode = True


class FirstHalf1Base(BaseModel):
    PHASE: Optional[str]
    TARIFF: Optional[str]
    POWER_SUSCRIBED: Optional[str]
    TENSION: Optional[str]
    DIVISION: Optional[str]
    ACTIVITY_CMS: Optional[str]
    REGION: Optional[str]
    AGENCY: Optional[str]
    payment_mode: Optional[str]
    IsActive: Optional[bool]
    Tenure_Days: Optional[int]
    Avg_Consumption: Optional[float]
    Max_Consumption: Optional[float]
    Total_Payment: Optional[float]
    Spike_Count: Optional[int]
    Has_inspected: Optional[bool]
    inspection_count: Optional[int]
    fraud_found_category: Optional[str]
    Found_fraud: Optional[bool]

class FirstHalf1Create(FirstHalf1Base):
    SERVICE_NBR: str

class FirstHalf1Response(FirstHalf1Base):
    SERVICE_NBR: str

    class Config:
        orm_mode = True

from pydantic import BaseModel

class FraudListSchema(BaseModel):
    service_nbr: int

    class Config:
        orm_mode = True

class FirstHalfDataSchema(BaseModel):
    service_nbr: str
    phase: str
    tariff: str
    power_subscribed: float
    tension: str
    division: str
    activity_cms: Optional[str]
    region: str
    agency: str
    payment_mode: str
    is_active: int
    tenure_days: int
    avg_consumption: float
    max_consumption: float
    total_payment: float
    spike_count: int
    has_inspected: int
    inspection_count: int
    fraud_found_category: int
    found_fraud: int

    class Config:
        orm_mode = True

from pydantic import BaseModel
from typing import Optional

class ModelSchema(BaseModel):
    PHASE: str
    TARIFF: str
    POWER_SUSCRIBED: float
    TENSION: str
    DIVISION: str
    ACTIVITY_CMS: Optional[str]
    REGION: str
    AGENCY: str
    payment_mode: str
    IsActive: int
    Tenure_Days: int
    Avg_Consumption: float
    Max_Consumption: float
    Total_Payment: float
    Spike_Count: int
    Has_inspected: int
    inspection_count: int
    fraud_found_category: int
    Found_fraud: int
    SERVICE_NBR: int

    class Config:
        orm_mode = True
