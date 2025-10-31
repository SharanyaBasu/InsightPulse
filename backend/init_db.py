from db import Base, engine
from models import MarketData, MacroData

print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("Done.")
