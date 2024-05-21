from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import random
from config import DATABASE_URL

Base = declarative_base()

# SQLite 데이터베이스 연결 설정
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 테이블 정의
class Resource(Base):
    __tablename__ = 'resources'
    id = Column(Integer, primary_key=True, index=True)
    resource_name = Column(String, index=True)
    system_name = Column(String, index=True)
    amount = Column(Integer)

# account 테이블 정의
class Account(Base):
    __tablename__ = 'account'
    id = Column(Integer, primary_key=True, index=True)
    account = Column(String, unique=True, index=True)
    id_no = Column(String)
    balance = Column(Integer, default=0)
    user_name = Column(String, nullable=True)

# transaction_log 테이블 정의
class TransactionLog(Base):
    __tablename__ = 'transaction_log'
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.now())
    kind = Column(String)
    account = Column(String)
    amount = Column(Integer, default=0)
    to_account = Column(String, nullable=True)
    to_bank = Column(String, nullable=True)
    
# 테이블 생성
Base.metadata.create_all(bind=engine)

legacy_map = {}

class Legacy:

    @classmethod
    def get_legacy_name(cls):
        return next((key for key, value in legacy_map.items() if value == cls), "No Value")

class AssetManager(Legacy):

    # private method
    @classmethod
    def __add_resource(cls, resource_name, system_name, added_resource):
        db = SessionLocal()
        try:
            resource = db.query(Resource).filter_by(resource_name=resource_name, system_name=system_name).first()
            if resource:
                resource.amount += added_resource
            else:
                resource = Resource(resource_name=resource_name, system_name=system_name, amount=added_resource)
                db.add(resource)
            db.commit()
            return {resource_name: resource.amount}
        finally:
            db.close()

    # table 전체 조회
    @classmethod
    def get_resource_table(cls):
        db = SessionLocal()
        try:
            resources = db.query(Resource).all()
            resource_table = {}
            for resource in resources:
                if resource.resource_name not in resource_table:
                    resource_table[resource.resource_name] = {}
                resource_table[resource.resource_name][resource.system_name] = resource.amount
            return resource_table
        finally:
            db.close()

    # 특정 system_name, 자원의 양을 조회
    @classmethod
    def get_resource_status(cls, system_name):
        db = SessionLocal()
        try:
            resources = db.query(Resource).filter_by(system_name=system_name).all()
            return {resource.resource_name: resource.amount for resource in resources}
        finally:
            db.close()

    # 특정 system_name, 특정 자원의 량을 추가
    @classmethod
    def put_resource(cls, resource_name, system_name, added_resource):
        return cls.__add_resource(resource_name, system_name, added_resource)

    # 자원을 타 system_name에게 옮김
    @classmethod
    def move_resource(cls, resource_name, sender_name, receiver_name, added_resource):        
        return cls.__add_resource(resource_name, sender_name, -added_resource), \
               cls.__add_resource(resource_name, receiver_name, added_resource)

class InternetBanking(Legacy):

    @classmethod
    def __validate_account(cls, account):
        db = SessionLocal()
        try:
            acc = db.query(Account).filter_by(account=account).first()
            return acc is not None
        finally:
            db.close()

    @classmethod
    def create_account(cls, id_no, user_name):
        db = SessionLocal()
        try:
            account = str(random.randint(10000000, 99999999))
            new_account = Account(account=account, user_name=user_name, id_no=id_no)
            db.add(new_account)
            db.commit()
            return {"success": f"Account {account} created for user {user_name}"}
        finally:
            db.close()

    @classmethod
    def deposit(cls, account, amount):
        if not cls.__validate_account(account):
            return {"error": "Account does not exist"}
        
        db = SessionLocal()
        try:
            # Transaction log 기록
            transaction = TransactionLog(
                date=datetime.now(),
                kind='deposit',
                account=account,
                amount=amount
            )
            db.add(transaction)
            
            # Account balance 업데이트
            acc = db.query(Account).filter_by(account=account).first()
            acc.balance += amount
            
            db.commit()
            return {"success": f"Deposited {amount} to {account}"}
        finally:
            db.close()

    @classmethod
    def withdraw(cls, account, amount):
        if not cls.__validate_account(account):
            return {"error": "Account does not exist"}
        
        db = SessionLocal()
        try:
            # Transaction log 기록
            transaction = TransactionLog(
                date=datetime.now(),
                kind='withdraw',
                account=account,
                amount=amount,
            )
            db.add(transaction)
            
            # Account balance 업데이트
            acc = db.query(Account).filter_by(account=account).first()
            if acc.balance < amount:
                return {"error": "Insufficient funds"}
            acc.balance -= amount
            
            db.commit()
            return {"success": f"Withdrew {amount} from {account}"}
        finally:
            db.close()

    @classmethod
    def external_transfer(cls, account, amount, to_bank, to_account):
        if not cls.__validate_account(account):
            return {"error": "Account does not exist"}
        
        db = SessionLocal()
        try:
            # Transaction log 기록
            transaction = TransactionLog(
                date=datetime.now(),
                kind='transfer',
                account=account,
                amount=amount,
                to_account=to_account,
                to_bank=to_bank
            )
            db.add(transaction)
            
            # Account balance 업데이트
            acc = db.query(Account).filter_by(account=account).first()
            if acc.balance < amount:
                return {"error": "Insufficient funds"}
            acc.balance -= amount
            
            db.commit()
            return {"success": f"Transferred {amount} from {account} to {to_account} at {to_bank}"}
        finally:
            db.close()

class CoreBanking(Legacy):
    pass

class Channel(Legacy):
    pass

legacy_map = {
    '자산관리': AssetManager ,
    '코어뱅킹': CoreBanking ,
    '채널관리': Channel ,
    '인터넷뱅킹': InternetBanking ,
}

# 테스트
if __name__ == "__main__":
    AssetManager.put_resource('memory', 'system1', 100)
    AssetManager.put_resource('cpu', 'system1', 50)
    AssetManager.put_resource('disk', 'system2', 200)

    print(AssetManager.get_resource_table())
    print(AssetManager.get_resource_status('system1'))
    AssetManager.move_resource('memory', 'system1', 'system2', 50)
    print(AssetManager.get_resource_table())
    print(AssetManager.get_resource_status('system2'))

    # 예제 계좌 생성
    print(InternetBanking.create_account("123456", "John Doe"))

    # 테스트 실행
    print(InternetBanking.deposit("123456", 500))
    print(InternetBanking.withdraw("123456", 200))
    print(InternetBanking.external_transfer("123456", 100, "OtherBank", "654321"))
