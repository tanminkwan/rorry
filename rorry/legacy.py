class Legacy:

    # 누가 어떤 자원을 얼만큼 가지고 있는지를 관리하는 database : 
    # memory, cpu, disk 3가지의 resource 존재. 
    # user_id를 key, 숫자값의 value를 가짐
    __resource_database = {"memory":{}, "cpu":{}, "disk":{}}

    # private method
    @classmethod
    def __add_resource(cls, resource_name, system_name, added_resource):
        if cls.__resource_database[resource_name].get(system_name):
            cls.__resource_database[resource_name][system_name] += added_resource
        else:
            cls.__resource_database[resource_name][system_name] = added_resource
        return cls.__resource_database[resource_name][system_name]

    # table 전체 조회
    @classmethod
    def get_resource_table(cls):
        return cls.__resource_database

    # 특정 system_name, 자원의 양을 조회
    @classmethod
    def get_resource_status(cls, system_name):
        db = cls.__resource_database
        return { key:db[key][system_name] for key in db if db[key].get(system_name) }

    # 특정 system_name, 특정 자원의 량을 추가
    @classmethod
    def put_resource(cls, resource_name, system_name, added_resource):
        return cls.__add_resource(resource_name, system_name,added_resource)

    # 자원을 타 system_name에게 옮김
    @classmethod
    def move_resource(cls, resource_name, sender_name, receiver_name, added_resource):        
        return cls.__add_resource(resource_name, sender_name, -added_resource), \
            cls.__add_resource(resource_name, receiver_name, added_resource)