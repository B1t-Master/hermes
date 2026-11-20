"""OpenAI-compatible tool schemas offered to the LLM.

Flight tools go through the ``hermes-flights`` MCP server; rebooking and
booking flows go through ``hermes-ops``. The schemas are part of the public
contract with the model, so their names and arguments must stay stable.
"""

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_flight_status",
            "description": (
                "Look up the current status of one Kenya Airways flight "
                "(on time, delayed, cancelled) and its scheduled times."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_no": {
                        "type": "string",
                        "description": 'Flight number such as "KQ100".',
                    },
                    "date": {
                        "type": "string",
                        "description": "Travel date YYYY-MM-DD; defaults to today.",
                    },
                },
                "required": ["flight_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_delays",
            "description": (
                "List Kenya Airways flights delayed by 15 minutes or more, "
                "optionally filtered to one route."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "route": {
                        "type": "string",
                        "description": 'Optional route "NBO-LHR" or a single airport code.',
                    },
                    "date": {
                        "type": "string",
                        "description": "Travel date YYYY-MM-DD; defaults to today.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rebook_flight",
            "description": (
                "Rebook a passenger onto an available flight. Requires the "
                "passenger's booking reference or e-ticket number and the new "
                "travel date. When no flight number is supplied, returns the "
                "available alternatives for the passenger to choose from. "
                "When a flight number is supplied, confirms the rebooking. "
                "Emails a confirmation only when the booking has an email on "
                "file; ask for one if missing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_no": {
                        "type": "string",
                        "description": (
                            "Flight number chosen from the alternatives, e.g. "
                            '"KQ444". Omit to see the available alternatives first.'
                        ),
                    },
                    "ticket_number": {
                        "type": "string",
                        "description": (
                            "The passenger's booking reference or e-ticket number. "
                            "If the passenger has not provided it yet, ask for it "
                            "first instead of calling this tool."
                        ),
                    },
                    "new_date": {
                        "type": "string",
                        "description": (
                            "The passenger's new travel date, e.g. 'tomorrow', "
                            "'tomorrow afternoon', 'day after tomorrow' or "
                            "YYYY-MM-DD."
                        ),
                    },
                    "passenger_name": {
                        "type": "string",
                        "description": "Passenger name, when known.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the flight is being rebooked (delay, cancellation, schedule change).",
                    },
                },
                "required": ["ticket_number", "new_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_rebook",
            "description": (
                "Confirm a rebooking: update the existing booking with the new "
                "flight and date, then email the new e-ticket. No payment is "
                "needed because demo fares are constant for rebooks. Call this "
                "after rebook_flight returns the alternatives and the passenger "
                "has chosen one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_number": {
                        "type": "string",
                        "description": "The passenger's booking reference or e-ticket number.",
                    },
                    "new_date": {
                        "type": "string",
                        "description": "The new travel date, e.g. 'tomorrow' or YYYY-MM-DD.",
                    },
                    "flight_no": {
                        "type": "string",
                        "description": 'Flight number chosen from the alternatives, e.g. "KQ444".',
                    },
                    "email": {
                        "type": "string",
                        "description": "Contact email for the new e-ticket (optional).",
                    },
                    "passenger_name": {
                        "type": "string",
                        "description": "Passenger name, when known.",
                    },
                },
                "required": ["ticket_number", "new_date", "flight_no"],
            },
        },
    },
]


BOOKING_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": (
                "Search Kenya Airways flights for one route and travel date. "
                "Use this before create_booking; never invent flights or fares. "
                "Origin and destination may be city names or IATA codes, and "
                "the date may be natural language ('tomorrow', '12 December') "
                "or YYYY-MM-DD."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": 'Departure city or code, e.g. "Nairobi" or "NBO".',
                    },
                    "destination": {
                        "type": "string",
                        "description": 'Arrival city or code, e.g. "Mombasa" or "MBA".',
                    },
                    "date": {
                        "type": "string",
                        "description": "Travel date, e.g. 'tomorrow', '12 December' or '2026-12-12'.",
                    },
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_booking",
            "description": (
                "Create a held booking for a flight returned by search_flights. "
                "The reply renders an M-Pesa payment card that collects the "
                "passenger's number and starts the STK push, so never ask for "
                "the number in chat. Call this once you have a flight number "
                "and travel date. The result carries the PNR, fare, pending "
                "payment id, and an awaiting_phone flag."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_no": {
                        "type": "string",
                        "description": 'Flight number from search_flights, e.g. "KQ444".',
                    },
                    "date": {
                        "type": "string",
                        "description": "Travel date for the chosen flight.",
                    },
                    "passenger_name": {
                        "type": "string",
                        "description": "Passenger name, when known.",
                    },
                    "passenger_email": {
                        "type": "string",
                        "description": "Passenger email for e-ticket delivery (optional).",
                    },
                    "passenger_phone": {
                        "type": "string",
                        "description": "Passenger phone number for M-Pesa payment (e.g. 0791351504).",
                    },
                },
                "required": ["flight_no", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_booking_email",
            "description": (
                "Add a passenger email to a paid booking and issue/email the "
                "e-ticket. Manual/administrative use only: the passenger booking "
                "flow ends at the payment card, and the agent dashboard issues "
                "the e-ticket after payment succeeds. Do not call this during a "
                "passenger conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pnr": {
                        "type": "string",
                        "description": "Booking reference (PNR) from create_booking.",
                    },
                    "email": {
                        "type": "string",
                        "description": "Passenger email address.",
                    },
                    "passenger_name": {
                        "type": "string",
                        "description": "Passenger name, when known.",
                    },
                },
                "required": ["pnr", "email"],
            },
        },
    },
]
