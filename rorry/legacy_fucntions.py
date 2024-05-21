legacy_fucntions = {}

legacy_fucntions['자산관리'] = \
    [
        {
            "name": "get_resource_status",
            "description": "특정 시스템의 자원상태를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "system_name": {
                        "type": "string",
                        "description": "시스템 이름, 이름의 앞이나 뒤에 '시스템' 이라는 글자는 포함하지 않음",
                    }
                },
                "required": ["system_name"],
            },
        },
        {
            "name": "put_resource",
            "description": "특정 시스템의 자원의 개수(량)을 변경합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_name": {
                        "type": "string",
                        "description": "자원 이름",
                    },
                    "system_name": {
                        "type": "string",
                        "description": "시스템 이름, 이름의 앞이나 뒤에 '시스템' 이라는 글자는 포함하지 않음",
                    },
                    "added_resource": {
                        "type": "integer",
                        "description": "자원의 증가량 또는 감소량, 값이 음수인 경우 감소량",
                    },
                },
                "required": ["resource_name","system_name","added_resource"],
            },
        },
        {
            "name": "move_resource",
            "description": "한 시스템의 특정 자원을 다른 시스템으로 이동합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_name": {
                        "type": "string",
                        "description": "자원명칭",
                    },
                    "sender_name": {
                        "type": "string",
                        "description": "자원을 보내거나 빼앗기는 시스템의 이름, 이름의 앞이나 뒤에 '시스템' 이라는 글자는 포함하지 않음",
                    },
                    "receiver_name": {
                        "type": "string",
                        "description": "자원을 받거나 빼앗는 시스템의 이름, 이름의 앞이나 뒤에 '시스템' 이라는 글자는 포함하지 않음",
                    },
                    "added_resource": {
                        "type": "integer",
                        "description": "자원의 증가량 또는 감소량, 값이 음수인 경우 감소량",
                    },
                },
                "required": ["resource_name","sender_name","receiver_name","added_resource"],
            },
        },

    ]

legacy_fucntions['코어뱅킹'] = \
    [
        {
            "name": "deposit",
            "description": "특정 계좌에 금액을 입금합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "bank": {
                        "type": "string",
                        "description": "은행명",
                    },
                    "account": {
                        "type": "string",
                        "description": "은행계좌",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "입금할 금액, 값은 항상 0보다 큼",
                    },
                },
                "required": ["bank","account","amount"],
            },
        },
    ]

legacy_fucntions['인터넷뱅킹'] = \
    [
        {
            "name": "deposit",
            "description": "산업은행 특정 계좌에 금액을 입금합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_name": {
                        "type": "string",
                        "description": "예금주, 계좌 주인",
                    },
                    "account": {
                        "type": "string",
                        "description": "은행계좌, 계좌번호",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "입금할 금액, 값은 항상 0보다 큼",
                    },
                },
                "required": ["account","amount"],
            },
        },
        {
            "name": "withdraw",
            "description": "산업은행 특정 계좌에서 금액을 출금합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_name": {
                        "type": "string",
                        "description": "예금주, 계좌 주인",
                    },
                    "account": {
                        "type": "string",
                        "description": "은행계좌, 계좌번호",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "출금할 금액, 값은 항상 0보다 큼",
                    },
                },
                "required": ["account","amount"],
            },
        },
        {
            "name": "external_transfer",
            "description": "산업은행에서 다른 은행으로 금액을 이체합니다. 타행이체를 합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_name": {
                        "type": "string",
                        "description": "예금주, 계좌 주인",
                    },
                    "account": {
                        "type": "string",
                        "description": "은행계좌, 계좌번호",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "이체할 금액, 값은 항상 0보다 큼",
                    },
                    "bank_name": {
                        "type": "string",
                        "description": "은행명, 은행이름, 타은행, 산업은행이 아닌 은행",
                    },
                    "to_account": {
                        "type": "string",
                        "description": "이체 받을 은행계좌, 이체 받을 계좌번호",
                    },
                },
                "required": ["account","amount", "bank_name", "to_account"],
            },
        },
        {
            "name": "balance",
            "description": "산업은행 특정 계좌의 잔액을 조회합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "account": {
                        "type": "string",
                        "description": "은행계좌, 계좌번호",
                    },
                },
                "required": ["account"],
            },
        },
    ]