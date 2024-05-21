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