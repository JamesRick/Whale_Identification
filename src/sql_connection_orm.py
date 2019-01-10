from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, func
from sqlalchemy import Column, String, Integer
import gc
import hashlib

def get_connection(connection_str):

    engine = create_engine(connection_str, isolation_level='READ COMMITTED', connect_args={'connect_timeout': 500})

    Base = declarative_base()

    class Similar(Base):
        __tablename__ = 'similar'

        hash = Column(String(64), primary_key=True)
        image_1 = Column(String(64))
        image_2 = Column(String(64))
        label = Column(Integer)

        def __repr__(self):
            return "<Similar(hash='%s', image_1='%s', image_2='%s', label='%d')>" %
                    (self.hash, self.image_1, self.image_2, self.label)

    class Different(Base):
        __tablename__ = 'different'

        hash = Column(String(64), primary_key=True)
        image_1 = Column(String(64))
        image_2 = Column(String(64))
        label = Column(Integer)

    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()
    return Similar, Different, session

class SQLConnection:

    def __init__(self, connection_str):
        self.Similar, self.Different, self.session = get_session(connection_str)

    def __get_different_batch(num):
        return self.session.query(self.Different).order_by(func.rand()).limit(num).all()

    def __get_similar_batch(num):
        return self.session.query(self.Similar).order_by(func.rand()).limit(num).all()

    def __load_images(image_query_list):
        image_names = []
        labels = []

        for row in image_query_list:
            if row.image_1 not in image_dict.keys():
                img_1 = io.imread(os.path.join('datasets/images/train_256_256/', row.image_1))
                image_dict[row.image_1] = img_1
            if row.image_2 not in image_dict.keys():
                img_2 = io.imread(os.path.join('datasets/images/train_256_256/', row.image_2))
                image_dict[row.image_2] = img_2
            labels.append(int(row.label))
            image_names.append((row.image_1, row.image_2))

        gc.collect()
        return image_dict, image_names, labels

    def image_hash(image_tuple):
        return hashlib.sha256(str(image_tuple).encode()).hexdigest()

    def get_batch():
        similar_number = int(np.random.rand() * 32)
        different_number = 32 - similar_number

        different_query_list = __get_different_batch(different_num)
        similar_query_list = __get_similar_batch(similar_num)
        query_list = different_query_list + similar_query_list
        image_dict, image_names, labels = __load_images(query_list)
        return image_dict, image_names, labels

    def insert(hash, image_1, image_2, label):
        if label:
            row = self.Similar(hash=hash, image_1=image_1,
                image_2=image_2, label=label)
        else:
            row = self.Different(hash=hash, image_1=image_1,
                image_2=image_2, label=label)
        self.session.add(row)
        self.session.commit()
