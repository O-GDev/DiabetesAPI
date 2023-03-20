from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


SQLALCHEMY_DATABASE_URL = 'postgres://ecxswfcwcebesf:a14e3f71007bd9077a6aec2aecbbfb3fcb124c9e473194a48cf413d5cba6505c@ec2-3-93-160-246.compute-1.amazonaws.com:5432/d9prqspvakhrd8
'

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
