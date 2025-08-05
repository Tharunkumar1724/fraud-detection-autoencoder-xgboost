from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Connection string
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:root1@localhost:3306/cluster"

# Create engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()

# Dependency function for DB session
def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
