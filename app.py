"""
Automobile Service Booking Chatbot - Streamlit App (CA4)
----------------------------------------------------------
Same rule-based intent/entity logic as the CA3 Colab prototype, wrapped in
a tabbed Streamlit interface (chat, catalogue, booking form, reviews). Run with:

    streamlit run app.py
"""

import datetime
import re

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# 1. DOMAIN REFERENCE DATA (small catalogue / lookup tables)
# ---------------------------------------------------------------------------

ALL_VEHICLE_TYPES = ["car", "suv", "sedan", "hatchback", "bike", "scooter"]

# Product catalogue: each service is a "product" with structured attributes,
# not just a name and description, so it can be searched, filtered, and
# looked up in detail.
SERVICES = {
    "general service": {
        "desc": "Full inspection, oil & filter change, and safety check.",
        "price": 1200, "duration_mins": 90, "vehicles": ALL_VEHICLE_TYPES,
    },
    "oil change": {
        "desc": "Engine oil and oil filter replacement.",
        "price": 800, "duration_mins": 30, "vehicles": ALL_VEHICLE_TYPES,
    },
    "car wash": {
        "desc": "Exterior wash and interior vacuuming.",
        "price": 300, "duration_mins": 20, "vehicles": ALL_VEHICLE_TYPES,
    },
    "brake repair": {
        "desc": "Brake pad inspection and replacement.",
        "price": 1500, "duration_mins": 60, "vehicles": ALL_VEHICLE_TYPES,
    },
    "ac service": {
        "desc": "AC gas top-up and cooling system check.",
        "price": 1000, "duration_mins": 45,
        "vehicles": ["car", "suv", "sedan", "hatchback"],  # bikes/scooters have no AC
    },
    "tyre replacement": {
        "desc": "Tyre replacement/rotation and balancing.",
        "price": 2500, "duration_mins": 40, "vehicles": ALL_VEHICLE_TYPES,
    },
    "battery check": {
        "desc": "Battery health check and replacement if needed.",
        "price": 200, "duration_mins": 15, "vehicles": ALL_VEHICLE_TYPES,
    },
}

LOCATIONS = ["whitefield", "koramangala", "hsr layout", "mg road", "indiranagar"]


ACRONYMS = {"Ac": "AC", "Hsr": "HSR", "Mg": "MG", "Suv": "SUV"}


def display_name(text):
    """Pretty name for display. str.title() would turn 'ac service' into 'Ac Service'
    and 'hsr layout' into 'Hsr Layout', so known acronyms are restored word by word."""
    return " ".join(ACRONYMS.get(w, w) for w in text.title().split())

VEHICLE_KEYWORDS = ["car", "bike", "suv", "sedan", "hatchback", "scooter",
                     "honda", "maruti", "hyundai", "royal enfield", "tvs", "toyota"]

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

TIME_SLOTS = ["morning", "afternoon", "evening"]
# Same labels the booking form offers, so chat and form bookings look identical.
TIME_SLOT_OPTIONS = ["Morning (9 AM - 12 PM)", "Afternoon (12 PM - 4 PM)", "Evening (4 PM - 7 PM)"]

# Synthetic customer reviews, each tied to a service and a branch so they
# can be filtered either way. Sentiment is NOT pre-labeled here; it is
# computed at runtime by the lexicon-based analyzer below.
REVIEWS = [
    {"location": "koramangala", "service": "oil change", "text": "Quick and professional oil change, in and out in 20 minutes. Very happy."},
    {"location": "koramangala", "service": "general service", "text": "Staff were friendly and the car came back spotless. Highly recommend."},
    {"location": "whitefield", "service": "brake repair", "text": "Brake repair took much longer than promised and the staff were rude about the delay."},
    {"location": "whitefield", "service": "car wash", "text": "Decent wash, nothing special, a bit overpriced for what it is."},
    {"location": "hsr layout", "service": "ac service", "text": "AC is finally cold again, excellent work and very affordable too."},
    {"location": "hsr layout", "service": "tyre replacement", "text": "Terrible experience, the new tyre had a slow puncture within a week."},
    {"location": "mg road", "service": "battery check", "text": "Fast, helpful, and honest about not needing a replacement. Great service."},
    {"location": "mg road", "service": "general service", "text": "Average service, the waiting area was dirty and the wait was longer than expected."},
    {"location": "indiranagar", "service": "oil change", "text": "Smooth booking process and the mechanic explained everything clearly. Satisfied."},
    {"location": "indiranagar", "service": "brake repair", "text": "Good job on the brakes but the billing was confusing and took forever."},
    {"location": "koramangala", "service": "tyre replacement", "text": "Excellent and efficient, the team was polite and the price was fair."},
    {"location": "koramangala", "service": "ac service", "text": "Disappointed, the AC issue came back within days and they blamed the weather."},
    {"location": "whitefield", "service": "battery check", "text": "Quick check, no pressure to buy a new battery. Reliable and honest."},
    {"location": "whitefield", "service": "general service", "text": "Worst experience, my car was returned with a scratch that wasn't there before."},
    {"location": "hsr layout", "service": "car wash", "text": "Clean, affordable, and fast. My car looks amazing, will come back."},
    {"location": "mg road", "service": "oil change", "text": "Nothing wrong with it, service was fine, nothing exceptional either way."},
    {"location": "indiranagar", "service": "tyre replacement", "text": "Great value and the staff were very professional throughout."},
    {"location": "indiranagar", "service": "ac service", "text": "Poor communication, I was not informed about the extra charges until pickup."},
]

POSITIVE_WORDS = {
    "great", "excellent", "good", "amazing", "fantastic", "friendly", "quick",
    "fast", "professional", "clean", "affordable", "smooth", "helpful",
    "satisfied", "happy", "polite", "efficient", "reliable", "recommend",
    "perfect", "honest", "fair", "outstanding", "spotless", "value",
}

NEGATIVE_WORDS = {
    "bad", "poor", "slow", "rude", "expensive", "disappointed", "terrible",
    "awful", "unprofessional", "delay", "delayed", "issue", "problem",
    "dirty", "waste", "horrible", "worst", "unhappy", "complain",
    "complaint", "scratch", "confusing", "overpriced", "puncture",
}

INTENT_KEYWORDS = {
    # Order matters: more specific / conflicting intents are checked first.
    # e.g. "cancel my appointment" contains the word "appointment" but must
    # be classified as CANCELLATION, not APPOINTMENT.
    "GREETING": ["hi", "hello", "hey", "good morning", "good afternoon"],
    "CANCELLATION": ["cancel", "reschedule"],
    "GOODBYE": ["bye", "goodbye", "exit", "quit", "thank you", "that's all"],
    "SERVICE_TYPE": ["service type", "services", "what services", "types of service", "do you do"],
    "AVAILABILITY": ["available", "availability", "slot", "open on", "opening time", "timings",
                      "what time", "do you open", "are you open"],
    "LOCATION": ["location", "workshop", "branch", "where is", "nearest"],
    "APPOINTMENT": ["book", "appointment", "schedule", "book a service"],
    "REVIEWS": ["review", "reviews", "feedback", "rating", "ratings", "what do people say"],
}

YES_WORDS = ["yes", "yeah", "yep", "sure", "please do", "ok", "okay"]
NO_WORDS = ["no", "nope", "not now", "nah", "no thanks"]


# ---------------------------------------------------------------------------
# 2. INTENT DETECTION
# ---------------------------------------------------------------------------

def keyword_intent(user_text):
    """Return the first intent whose keyword appears as a whole word, else None."""
    text = user_text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                return intent
    return None


def detect_intent(user_text):
    """Return the matched intent name, or 'UNKNOWN' if nothing matches.

    Keyword matching uses word boundaries, not plain substring checks, so a
    short keyword like "hi" cannot accidentally match inside an unrelated
    word like "anything" (which contains the letters "h" "i" in sequence).
    """
    intent = keyword_intent(user_text)
    if intent:
        return intent

    text = user_text.lower()

    # Fallback 1: naming a specific service directly still counts as SERVICE_TYPE,
    # even without one of the keywords above (e.g. "tell me about brake repair").
    for service_name in SERVICES:
        if re.search(r"\b" + re.escape(service_name) + r"\b", text):
            return "SERVICE_TYPE"

    # Fallback 2: a price filter or a recognizable catalogue search also counts
    # (e.g. "services under 500", "do you have anything for tyres").
    if extract_price_constraint(user_text):
        return "SERVICE_TYPE"
    if search_services(user_text):
        return "SERVICE_TYPE"

    return "UNKNOWN"


def is_affirmative(text):
    text = text.lower()
    return any(re.search(r"\b" + re.escape(word) + r"\b", text) for word in YES_WORDS)


def is_negative(text):
    text = text.lower()
    return any(re.search(r"\b" + re.escape(word) + r"\b", text) for word in NO_WORDS)


# ---------------------------------------------------------------------------
# 3. ENTITY EXTRACTION
# ---------------------------------------------------------------------------

def extract_vehicle(text):
    text = text.lower()
    for kw in VEHICLE_KEYWORDS:
        if kw in text:
            return kw
    return None


def extract_service_type(text):
    text = text.lower()
    for service in SERVICES:
        if service in text:
            return service
    return None


def extract_location(text):
    text = text.lower()
    for loc in LOCATIONS:
        if loc in text:
            return loc
    return None


def extract_date(text):
    text = text.lower()
    if "today" in text:
        return "today"
    if "tomorrow" in text:
        return "tomorrow"
    for day in WEEKDAYS:
        if day in text:
            return day.capitalize()
    match = re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text)
    if match:
        return match.group()
    return None


def extract_appointment_time(text):
    """Read a time slot from chat ("morning", "3 pm", "16:30") and return the same
    label the booking form uses, e.g. 'Morning (9 AM - 12 PM)'. Returns None if no
    slot is found or the time is outside opening hours (9 AM - 7 PM)."""
    text = text.lower()
    for i, slot in enumerate(TIME_SLOTS):
        if re.search(r"\b" + slot + r"\b", text):
            return TIME_SLOT_OPTIONS[i]
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s?(am|pm)\b|\b(\d{1,2}):(\d{2})\b", text)
    if not match:
        return None
    if match.group(1):
        hour, meridian = int(match.group(1)), match.group(3)
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if meridian == "pm" else 0)
    else:
        hour = int(match.group(4))
        if not 0 <= hour <= 23:
            return None
        if hour < 9:          # "3:30" with no am/pm: nobody means 3:30 in the morning
            hour += 12
    if 9 <= hour < 12:
        return TIME_SLOT_OPTIONS[0]
    if 12 <= hour < 16:
        return TIME_SLOT_OPTIONS[1]
    if 16 <= hour < 19:
        return TIME_SLOT_OPTIONS[2]
    return None


def _has_word(text, words):
    return any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in words)


def extract_name(text):
    """Pick a customer name out of a chat reply ("Sam Kumar", "my name is sam").
    Returns None for anything that looks like a question, a command, or another
    answer (a date, service, branch...), so those never get saved as a name."""
    text = text.strip()
    explicit = re.search(r"\b(?:my name is|name is|i am|i'm|this is|it's|its)\s+(.+)", text, re.I)
    if explicit:
        text = explicit.group(1)
    elif keyword_intent(text):
        return None
    text = re.sub(r"\+?\d[\d\s\-]*", " ", text)           # drop a phone number sent in the same message
    text = re.sub(r"^[\s,.:;-]+|[\s,.:;-]+$", "", text)
    text = re.sub(r"\s+(and|phone|number|mobile|contact)\b.*$", "", text, flags=re.I)
    if not text or "?" in text or not re.fullmatch(r"[A-Za-z][A-Za-z.'\- ]{1,39}", text):
        return None
    words = text.lower().split()
    if len(words) > 4 or is_affirmative(text) or is_negative(text):
        return None
    question_starters = {"what", "how", "why", "where", "when", "which", "who", "do", "does",
                         "can", "could", "is", "are", "show", "tell", "list", "give", "price", "cost"}
    if words[0] in question_starters:
        return None
    others = (VEHICLE_KEYWORDS + list(SERVICES) + LOCATIONS + WEEKDAYS + TIME_SLOTS
              + ["today", "tomorrow"])
    if _has_word(text.lower(), others):
        return None
    return text.title() if text in (text.lower(), text.upper()) else text


def extract_phone(text):
    """Return a 10-digit phone number (same rule as the booking form), else None.
    A leading +91 / 91 country code is accepted and dropped."""
    digits = re.sub(r"\D", "", text)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    return digits if len(digits) == 10 else None


def is_sunday(date_text):
    """True if a chat date ('sunday', 'today', 'tomorrow', '12/10/2026') falls on a
    Sunday. The branches are closed on Sundays, same as the form enforces."""
    text = date_text.lower()
    today = datetime.date.today()
    if text == "sunday":
        return True
    if text == "today":
        return today.weekday() == 6
    if text == "tomorrow":
        return (today + datetime.timedelta(days=1)).weekday() == 6
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?", text)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else today.year
        if year < 100:
            year += 2000
        try:
            return datetime.date(year, month, day).weekday() == 6
        except ValueError:
            return False
    return False


def extract_price_constraint(text):
    """Look for a price filter like 'under 1000', 'below Rs.800', 'between 500 and 1000'."""
    text = text.lower()
    m = re.search(r"between\s*(?:rs\.?|₹)?\s*(\d+)\s*(?:and|-)\s*(?:rs\.?|₹)?\s*(\d+)", text)
    if m:
        return ("between", int(m.group(1)), int(m.group(2)))
    m = re.search(r"(?:under|below|less than|cheaper than)\s*(?:rs\.?|₹)?\s*(\d+)", text)
    if m:
        return ("under", int(m.group(1)))
    m = re.search(r"(?:above|over|more than|greater than)\s*(?:rs\.?|₹)?\s*(\d+)", text)
    if m:
        return ("above", int(m.group(1)))
    return None


def price_matches(price, constraint):
    kind = constraint[0]
    if kind == "under":
        return price < constraint[1]
    if kind == "above":
        return price > constraint[1]
    if kind == "between":
        return constraint[1] <= price <= constraint[2]
    return True


STOPWORDS = {"do", "you", "have", "any", "for", "the", "a", "an", "of", "is", "are",
             "what", "which", "show", "me", "services", "service", "offer", "offers",
             "search", "find", "looking", "want", "need", "something", "with", "and"}


def search_services(text):
    """Loose keyword search across service names and descriptions (simple plural
    handling: a trailing 's' on a search word is also tried without the 's')."""
    text = text.lower()
    words = re.findall(r"[a-z]+", text)
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]
    matches = []
    for name, info in SERVICES.items():
        haystack = name + " " + info["desc"].lower()
        for kw in keywords:
            variants = {kw}
            if kw.endswith("s") and len(kw) > 3:
                variants.add(kw[:-1])
            if any(v in haystack for v in variants):
                matches.append(name)
                break
    return matches


def analyze_sentiment(text):
    """Simple lexicon-based sentiment: count positive vs negative word matches."""
    words = re.findall(r"[a-z']+", text.lower())
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    return "Neutral"


# ---------------------------------------------------------------------------
# 4. CONVERSATION STATE HELPERS
#    The state dict is passed in explicitly (st.session_state.conversation_state)
#    rather than kept as a module-level global, so each browser session/user
#    gets its own independent conversation when deployed.
# ---------------------------------------------------------------------------

def new_state():
    return {
        "active_flow": None,
        "slots": {},
        "pending_confirm": None,
        "pending_data": {},
    }


def reset_state(state):
    state["active_flow"] = None
    state["slots"] = {}
    state["pending_confirm"] = None
    state["pending_data"] = {}


# ---------------------------------------------------------------------------
# 5. MULTI-TURN FLOW HANDLERS
# ---------------------------------------------------------------------------

def handle_appointment_flow(user_text, state):
    """Multi-turn Flow 1: Vehicle -> Service -> Date -> Time slot -> Location
    -> Name -> Phone -> Confirm. Collects the same details as the booking form."""
    slots = state["slots"]
    filled_now = False   # True once this message has answered a question; stops one reply
                         # (e.g. "whitefield") from also being read as the customer's name

    if "vehicle" not in slots:
        vehicle = extract_vehicle(user_text)
        if vehicle:
            slots["vehicle"] = vehicle
            filled_now = True
        else:
            return "Sure, let's book your service. First, what vehicle do you have (car/bike/model)?"

    if "service" not in slots:
        service = extract_service_type(user_text)
        if service:
            slots["service"] = service
            filled_now = True
        else:
            return f"Got it, a {slots['vehicle']}. Which service would you like? (e.g., " \
                   f"{', '.join(list(SERVICES.keys())[:3])} ...)"

    if "date" in slots and is_sunday(slots["date"]):
        slots.pop("date")    # e.g. a date carried over from an availability check
        return "Our branches are closed on Sundays. Please pick another date."

    if "date" not in slots:
        date = extract_date(user_text)
        if date and is_sunday(date):
            return "Our branches are closed on Sundays. Please pick another date."
        if date:
            slots["date"] = date
            filled_now = True
            if "time" not in slots:          # "tomorrow morning" answers both questions
                time_slot = extract_appointment_time(user_text)
                if time_slot:
                    slots["time"] = time_slot
        else:
            return f"Noted: {slots['service']} for your {slots['vehicle']}. " \
                   f"What date would you like to bring it in?"

    if "time" not in slots:
        time_slot = extract_appointment_time(user_text) if not filled_now else None
        if time_slot:
            slots["time"] = time_slot
            filled_now = True
        elif filled_now:
            return "Which time slot works for you? " + ", ".join(TIME_SLOT_OPTIONS) + "."
        else:
            return "Please pick a time slot: " + ", ".join(TIME_SLOT_OPTIONS) + ". " \
                   "(We're open 9 AM - 7 PM.)"

    if "location" not in slots:
        location = extract_location(user_text)
        if location:
            slots["location"] = location
            filled_now = True
        else:
            return f"Which branch would you prefer? Options: {', '.join(LOCATIONS)}."

    if "name" not in slots:
        name = extract_name(user_text) if not filled_now else None
        if name:
            slots["name"] = name
            filled_now = True
        elif filled_now:
            return "Almost done. May I have your name for the booking?"
        else:
            return "Sorry, I didn't catch that. What name should I put the booking under?"

    if "phone" not in slots:
        phone = extract_phone(user_text)
        if phone:
            slots["phone"] = phone
        elif filled_now:
            return f"Thanks, {slots['name']}. What's your 10-digit phone number?"
        else:
            return "That doesn't look like a valid number. Please enter exactly 10 digits."

    summary = (f"Booking confirmed for {slots['name']}! Phone: {slots['phone']}, "
               f"Vehicle: {slots['vehicle']}, Service: {slots['service']}, "
               f"Date: {slots['date']}, Time: {slots['time']}, "
               f"Location: {display_name(slots['location'])}. Anything else?")
    booking = dict(slots)
    reset_state(state)
    state["pending_confirm"] = "ANYTHING_ELSE"
    state["completed_booking"] = booking   # the UI picks this up and lists it under My Bookings
    return summary


def handle_availability_flow(user_text, state):
    """Multi-turn Flow 2: Date/time -> respond with open slots."""
    slots = state["slots"]

    if "date" not in slots:
        date = extract_date(user_text)
        if date:
            slots["date"] = date
        else:
            return "Sure, which date would you like to check availability for?"

    date = slots["date"]
    reset_state(state)
    state["pending_confirm"] = "BOOK_FROM_AVAILABILITY"
    state["pending_data"] = {"date": date}
    return f"We have morning and evening slots open on {date}. Would you like to book one?"


def extract_booking_ref(text):
    """How the user points at a booking: its number ('#3', 'booking 3', or just '3')
    or a date ('monday', 'tomorrow', '13/10'). Returns a (kind, value) pair or None."""
    t = text.lower().strip()
    m = re.search(r"(?:#|\bbooking\s*(?:number|no\.?|id)?\s*|\bid\s*)(\d+)\b", t) or re.fullmatch(r"(\d{1,4})", t)
    if m:
        return ("id", int(m.group(1)))
    date = extract_date(text)
    return ("date", date) if date else None


def booking_matches_date(ref, booking_date):
    """Chat bookings store the date as typed ('Monday', 'tomorrow'); form bookings store
    'Mon 13 Oct 2026'. Both are compared against what the user said."""
    ref, stored = ref.lower(), booking_date.lower()
    if ref == stored:
        return True
    today = datetime.date.today()
    fmt = "%a %d %b %Y"
    if ref == "today":
        return stored == today.strftime(fmt).lower()
    if ref == "tomorrow":
        return stored == (today + datetime.timedelta(days=1)).strftime(fmt).lower()
    if ref in WEEKDAYS:
        return stored.startswith(ref[:3])
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?", ref)
    if m:
        year = int(m.group(3)) if m.group(3) else today.year
        year += 2000 if year < 100 else 0
        try:
            return stored == datetime.date(year, int(m.group(2)), int(m.group(1))).strftime(fmt).lower()
        except ValueError:
            return False
    return False


def describe_booking(b):
    return (f"#{b['id']} {display_name(b['service'])} for your {b['vehicle']} on {b['date']} "
            f"({b['time']}) at the {display_name(b['branch'])} branch")


def handle_cancellation_flow(user_text, state):
    """Multi-turn Flow 3: pick one of the user's real bookings (by number or date),
    ask for a yes/no, then flag it in state['cancelled_ids']. The UI removes it from
    My Bookings, so the Book Now tab and the sidebar count stay in sync."""
    bookings = state.get("bookings", [])
    if not bookings:
        reset_state(state)
        return "You don't have any bookings to cancel."

    ref = extract_booking_ref(user_text)
    matches = []
    if ref and ref[0] == "id":
        matches = [b for b in bookings if b["id"] == ref[1]]
    elif ref:
        matches = [b for b in bookings if booking_matches_date(ref[1], b["date"])]
    elif len(bookings) == 1:
        matches = bookings

    if len(matches) == 1:
        reset_state(state)
        state["pending_confirm"] = "CANCEL_BOOKING"
        state["pending_data"] = {"id": matches[0]["id"]}
        return f"Found {describe_booking(matches[0])}. Cancel it? (yes/no)"

    listing = "\n".join(f"- {describe_booking(b)}" for b in (matches or bookings))
    if len(matches) > 1:
        return f"More than one booking matches:\n{listing}\nWhich booking number should I cancel?"
    if ref:
        return f"I couldn't find that booking. Your bookings:\n{listing}\nWhich booking number should I cancel?"
    return f"Which booking would you like to cancel? Your bookings:\n{listing}\nReply with the booking number."


# ---------------------------------------------------------------------------
# 6. SINGLE-TURN RESPONSE HANDLERS
# ---------------------------------------------------------------------------

def handle_greeting():
    return "Hello! Welcome to AutoCare Service Bot. I can help with services, " \
           "bookings, availability, location, cancellations, or customer reviews. " \
           "How can I help you today?"


def format_service_line(name, info):
    vehicles = ", ".join(info["vehicles"])
    return (f"- {display_name(name)}: {info['desc']} "
            f"(₹{info['price']}, ~{info['duration_mins']} min, for: {vehicles})")


def handle_service_type(user_text):
    """Covers four related behaviours from one intent: full catalogue browsing,
    single-item detail lookup, keyword search, and price/vehicle filtering."""
    price_constraint = extract_price_constraint(user_text)
    vehicle = extract_vehicle(user_text)
    exact_service = extract_service_type(user_text)

    # 1. Specific service named with no filter -> show full details (product info)
    if exact_service and not price_constraint:
        info = SERVICES[exact_service]
        return f"{display_name(exact_service)} details:\n{format_service_line(exact_service, info)}"

    # 2. A price and/or vehicle filter was given -> filter the catalogue
    if price_constraint or vehicle:
        results = []
        for name, info in SERVICES.items():
            if price_constraint and not price_matches(info["price"], price_constraint):
                continue
            if vehicle and vehicle not in info["vehicles"]:
                continue
            results.append((name, info))
        if not results:
            return "No services match that filter. Try a different price range or vehicle type."
        lines = [format_service_line(n, i) for n, i in results]
        return "Here's what matches:\n" + "\n".join(lines)

    # 3. Loose keyword search (e.g. "do you have anything for tyres")
    found = search_services(user_text)
    if found and len(found) < len(SERVICES):
        lines = [format_service_line(n, SERVICES[n]) for n in found]
        return "Here's what I found:\n" + "\n".join(lines)

    # 4. Fallback: show the full catalogue
    lines = [format_service_line(n, i) for n, i in SERVICES.items()]
    return "Here are the services we offer:\n" + "\n".join(lines)


def handle_location(user_text):
    location = extract_location(user_text)
    if location:
        return f"Yes, we have a branch in {display_name(location)}. It's open 9 AM - 7 PM, Mon-Sat."
    return f"Our branches are located in: {', '.join(display_name(l) for l in LOCATIONS)}. " \
           "Which one are you asking about?"


def handle_reviews(user_text):
    service = extract_service_type(user_text)
    location = extract_location(user_text)

    filtered = REVIEWS
    if service:
        filtered = [r for r in filtered if r["service"] == service]
    if location:
        filtered = [r for r in filtered if r["location"] == location]

    note = ""
    if (service or location) and not filtered:
        note = "No reviews matched that exactly, so here are some recent reviews instead:\n"
        filtered = REVIEWS

    shown = filtered[:5]
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    lines = []
    for r in shown:
        sentiment = analyze_sentiment(r["text"])
        counts[sentiment] += 1
        lines.append(f"- [{sentiment}] ({display_name(r['location'])}, {display_name(r['service'])}): \"{r['text']}\"")

    summary = f"\n\nOut of {len(shown)} reviews shown: {counts['Positive']} positive, " \
              f"{counts['Negative']} negative, {counts['Neutral']} neutral."
    return note + "\n".join(lines) + summary


def handle_goodbye():
    return "Thank you for visiting AutoCare Service Bot. Have a great day! Goodbye."


def handle_unknown():
    return "Sorry, I didn't quite understand that. You can ask about services, " \
           "book an appointment, check availability, workshop location, cancel a booking, " \
           "or ask what customers are saying in reviews."


# ---------------------------------------------------------------------------
# 7. ROUTER
# ---------------------------------------------------------------------------

# For each multi-turn flow: the slots it collects, in order, and how to read each one.
FLOW_SLOTS = {
    "APPOINTMENT": [("vehicle", extract_vehicle), ("service", extract_service_type),
                    ("date", extract_date), ("time", extract_appointment_time),
                    ("location", extract_location), ("name", extract_name),
                    ("phone", extract_phone)],
    "AVAILABILITY": [("date", extract_date)],
    "CANCELLATION": [("booking", extract_booking_ref)],
}


def wants_to_leave_flow(user_text, state):
    """True when the user asks about something else while a flow is waiting for
    an answer (e.g. asks for reviews in the middle of a booking).

    A message that answers the question just asked keeps the flow going, even if
    it also contains a LOCATION or SERVICE_TYPE keyword ("the whitefield branch").
    Any other message with a keyword for a different intent interrupts the flow."""
    flow = state["active_flow"]
    other = keyword_intent(user_text)
    if other is None or other == flow:
        return False
    for slot, extractor in FLOW_SLOTS[flow]:
        if slot not in state["slots"]:
            if extractor(user_text):
                if other in ("LOCATION", "SERVICE_TYPE"):
                    return False
                # "morning slot please" contains the word "slot" (an AVAILABILITY keyword),
                # and "Hi, I'm Sam" starts with a greeting. Both are still answers.
                if slot in ("name", "phone"):
                    return False
                if slot == "time" and len(user_text.split()) <= 4:
                    return False
            break
    return True


def chatbot_response(user_text, state):
    """Wrapper around route_message: lets a new question interrupt a flow in
    progress, and tells the user when a booking was put on hold."""
    note = ""
    if state["active_flow"] and wants_to_leave_flow(user_text, state):
        if state["active_flow"] == "APPOINTMENT" and state["slots"]:
            note = "\n\n(Your booking is on hold. Say \"book a service\" to start it again.)"
        reset_state(state)
    reply = route_message(user_text, state)
    if note and state["active_flow"] is None:
        reply += note
    return reply


def route_message(user_text, state):
    """Central router: decides which handler responds, using the conversation
    state to keep multi-turn flows and yes/no follow-ups alive across turns."""

    if state["active_flow"] == "APPOINTMENT":
        return handle_appointment_flow(user_text, state)
    if state["active_flow"] == "AVAILABILITY":
        return handle_availability_flow(user_text, state)
    if state["active_flow"] == "CANCELLATION":
        return handle_cancellation_flow(user_text, state)

    if state["pending_confirm"] == "BOOK_FROM_AVAILABILITY":
        if is_affirmative(user_text):
            date = state["pending_data"].get("date")
            reset_state(state)
            state["active_flow"] = "APPOINTMENT"
            if date:
                state["slots"]["date"] = date
            return handle_appointment_flow(user_text, state)
        elif is_negative(user_text):
            reset_state(state)
            return "No problem! Let me know if there's anything else I can help with."
        else:
            reset_state(state)  # fall through to fresh intent detection below

    elif state["pending_confirm"] == "CANCEL_BOOKING":
        booking_id = state["pending_data"].get("id")
        if is_affirmative(user_text):
            still_there = any(b["id"] == booking_id for b in state.get("bookings", []))
            reset_state(state)
            if not still_there:
                return "That booking has already been cancelled."
            state["cancelled_ids"] = [booking_id]   # the UI removes it from My Bookings
            state["pending_confirm"] = "ANYTHING_ELSE"
            return f"Booking #{booking_id} has been cancelled. Let us know if you'd like to rebook. Anything else?"
        elif is_negative(user_text):
            reset_state(state)
            return "No problem, your booking is still on. Anything else I can help with?"
        else:
            reset_state(state)  # fall through to fresh intent detection below

    elif state["pending_confirm"] == "ANYTHING_ELSE":
        if is_affirmative(user_text):
            reset_state(state)
            return "Great - what would you like to do? (services, book appointment, availability, location, cancellation)"
        elif is_negative(user_text):
            reset_state(state)
            return handle_goodbye()
        else:
            reset_state(state)  # fall through to fresh intent detection below

    intent = detect_intent(user_text)

    if intent == "GREETING":
        return handle_greeting()
    elif intent == "SERVICE_TYPE":
        return handle_service_type(user_text)
    elif intent == "APPOINTMENT":
        state["active_flow"] = "APPOINTMENT"
        return handle_appointment_flow(user_text, state)
    elif intent == "AVAILABILITY":
        state["active_flow"] = "AVAILABILITY"
        return handle_availability_flow(user_text, state)
    elif intent == "LOCATION":
        return handle_location(user_text)
    elif intent == "CANCELLATION":
        state["active_flow"] = "CANCELLATION"
        return handle_cancellation_flow(user_text, state)
    elif intent == "REVIEWS":
        return handle_reviews(user_text)
    elif intent == "GOODBYE":
        return handle_goodbye()
    else:
        return handle_unknown()


# ---------------------------------------------------------------------------
# 8. HELPERS USED BY THE INTERFACE
#    These reuse the same catalogue, review and sentiment logic as the chatbot,
#    so the tabs and the chat always agree with each other.
# ---------------------------------------------------------------------------

SERVICE_ICONS = {
    "general service": "🧰", "oil change": "🛢️", "car wash": "🚿",
    "brake repair": "🛑", "ac service": "❄️", "tyre replacement": "🛞",
    "battery check": "🔋",
}

SENTIMENT_COLORS = {"Positive": "green", "Negative": "red", "Neutral": "gray"}

SORT_OPTIONS = ["Price: low to high", "Price: high to low", "Duration: shortest first", "Name (A-Z)"]

QUICK_QUESTIONS = [
    "What services do you offer",
    "Services under 500",
    "Show me some reviews",
    "I want to book a service",
    "What time do you open",
    "Where is your workshop",
]


def filter_catalogue(query, vehicle, price_range, sort_by):
    """Search + filter + sort the service catalogue (used by the Catalogue tab)."""
    names = list(SERVICES.keys())
    if query.strip():
        names = [n for n in search_services(query) if n in names]
    results = []
    for name in names:
        info = SERVICES[name]
        if vehicle != "Any" and vehicle not in info["vehicles"]:
            continue
        if not (price_range[0] <= info["price"] <= price_range[1]):
            continue
        results.append((name, info))
    if sort_by == "Price: low to high":
        results.sort(key=lambda r: r[1]["price"])
    elif sort_by == "Price: high to low":
        results.sort(key=lambda r: -r[1]["price"])
    elif sort_by == "Duration: shortest first":
        results.sort(key=lambda r: r[1]["duration_mins"])
    else:
        results.sort(key=lambda r: r[0])
    return results


def filter_reviews(service, location):
    """Return reviews (with computed sentiment) for an optional service/branch."""
    out = []
    for r in REVIEWS:
        if service != "All services" and r["service"] != service:
            continue
        if location != "All branches" and r["location"] != location:
            continue
        out.append({**r, "sentiment": analyze_sentiment(r["text"])})
    return out


def explain_sentiment(text):
    """Show which lexicon words drove the sentiment label (for the live analyzer)."""
    words = re.findall(r"[a-z']+", text.lower())
    pos = [w for w in words if w in POSITIVE_WORDS]
    neg = [w for w in words if w in NEGATIVE_WORDS]
    return pos, neg


def banner_svg():
    """Header illustration drawn in code, so no image files are needed.
    (Shown with st.image: st.html strips inline <svg> elements.)"""
    return """
<svg viewBox="0 0 900 220" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#1e3a8a"/>
      <stop offset="100%" stop-color="#0ea5e9"/>
    </linearGradient>
  </defs>
  <rect width="900" height="220" fill="url(#bg)"/>
  <circle cx="820" cy="40" r="26" fill="#fde68a" opacity="0.9"/>
  <rect y="180" width="900" height="40" fill="#0f172a"/>
  <line x1="0" y1="200" x2="900" y2="200" stroke="#fbbf24" stroke-width="4" stroke-dasharray="30 22"/>
  <g transform="translate(560,92)">
    <path d="M10 70 L30 30 Q40 15 60 15 L150 15 Q175 15 190 40 L215 55 Q230 60 230 75 L230 88 L10 88 Z" fill="#f8fafc"/>
    <path d="M52 30 L88 30 L88 55 L38 55 Z" fill="#7dd3fc"/>
    <path d="M98 30 L150 30 Q165 32 175 55 L98 55 Z" fill="#7dd3fc"/>
    <circle cx="65" cy="90" r="20" fill="#111827"/><circle cx="65" cy="90" r="9" fill="#9ca3af"/>
    <circle cx="180" cy="90" r="20" fill="#111827"/><circle cx="180" cy="90" r="9" fill="#9ca3af"/>
  </g>
  <text x="40" y="82" font-family="Segoe UI, Arial, sans-serif" font-size="44" font-weight="700" fill="#ffffff">AutoCare Service Bot</text>
  <text x="40" y="122" font-family="Segoe UI, Arial, sans-serif" font-size="20" fill="#e0f2fe">Book services, compare prices, read reviews.</text>
  <g font-family="Segoe UI, Arial, sans-serif" font-size="16" font-weight="600" fill="#0f172a">
    <rect x="40" y="144" width="130" height="30" rx="15" fill="#e0f2fe"/><text x="64" y="165">7 services</text>
    <rect x="182" y="144" width="130" height="30" rx="15" fill="#e0f2fe"/><text x="206" y="165">5 branches</text>
    <rect x="324" y="144" width="150" height="30" rx="15" fill="#e0f2fe"/><text x="344" y="165">Live reviews</text>
  </g>
</svg>
""".strip()


def steps_svg():
    """Small 'how it works' graphic."""
    return """
<svg viewBox="0 0 900 130" xmlns="http://www.w3.org/2000/svg">
  <rect width="900" height="130" rx="12" fill="#f1f5f9" stroke="#cbd5e1"/>
  <line x1="150" y1="50" x2="450" y2="50" stroke="#94a3b8" stroke-width="3"/>
  <line x1="450" y1="50" x2="750" y2="50" stroke="#94a3b8" stroke-width="3"/>
  <g fill="#1e3a8a"><circle cx="150" cy="50" r="26"/><circle cx="450" cy="50" r="26"/><circle cx="750" cy="50" r="26"/></g>
  <g font-family="Segoe UI, Arial, sans-serif" font-size="26" font-weight="700" fill="#ffffff" text-anchor="middle">
    <text x="150" y="59">1</text><text x="450" y="59">2</text><text x="750" y="59">3</text>
  </g>
  <g font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700" fill="#0f172a" text-anchor="middle">
    <text x="150" y="102">Ask</text><text x="450" y="102">Choose</text><text x="750" y="102">Confirm</text>
  </g>
  <g font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#475569" text-anchor="middle">
    <text x="150" y="122">Chat or browse services</text>
    <text x="450" y="122">Pick vehicle, service, date</text>
    <text x="750" y="122">Get a booking summary</text>
  </g>
</svg>
""".strip()


# ---------------------------------------------------------------------------
# 9. STREAMLIT UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AutoCare Service Bot", page_icon="🚗", layout="wide")

# --- Session state setup (runs once per browser session) ---
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = new_state()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": handle_greeting()}]
if "bookings" not in st.session_state:
    st.session_state.bookings = []
if "next_booking_id" not in st.session_state:
    st.session_state.next_booking_id = 1
if "form_version" not in st.session_state:
    st.session_state.form_version = 0   # bumping this clears the Book Now form
if "booking_notice" not in st.session_state:
    st.session_state.booking_notice = None


def add_booking(vehicle, service, date, branch, name="-", phone="-", time="-", source="Form"):
    st.session_state.bookings.append({
        "id": st.session_state.next_booking_id,
        "name": name, "phone": phone, "vehicle": vehicle, "service": service,
        "date": date, "time": time, "branch": branch, "source": source,
    })
    st.session_state.next_booking_id += 1


def process_message(text):
    """Send one user message through the chatbot and store both sides of the chat.
    If the chat just completed a booking, it is also added to 'My Bookings'."""
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.conversation_state["bookings"] = st.session_state.bookings   # so chat can see/cancel them
    response = chatbot_response(text, st.session_state.conversation_state)
    st.session_state.messages.append({"role": "assistant", "content": response})
    cancelled = st.session_state.conversation_state.pop("cancelled_ids", [])
    if cancelled:
        st.session_state.bookings = [b for b in st.session_state.bookings if b["id"] not in cancelled]
    done = st.session_state.conversation_state.pop("completed_booking", None)
    if done:
        add_booking(done["vehicle"], done["service"], done["date"], done["location"],
                    name=done["name"], phone=done["phone"], time=done["time"], source="Chat")


def reset_conversation():
    st.session_state.conversation_state = new_state()
    st.session_state.messages = [{"role": "assistant", "content": handle_greeting()}]


# --- Header image ---
st.image(banner_svg(), width="stretch")

# --- Sidebar: quick questions, branches, booking count, reset ---
with st.sidebar:
    st.header("🚗 AutoCare")
    st.caption("Quick questions (the answer appears in the Chat tab)")
    for q in QUICK_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"quick_{q}"):
            reset_state(st.session_state.conversation_state)   # a quick question never continues an old flow
            process_message(q)
    st.divider()
    st.subheader("📍 Branches")
    for loc in LOCATIONS:
        st.write(f"• {display_name(loc)}")
    st.caption("Open 9 AM - 7 PM, Mon-Sat")
    st.divider()
    bookings_slot = st.empty()   # filled at the end so the count is always current
    if st.button("🔄 New Conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

tab_chat, tab_catalogue, tab_book, tab_reviews = st.tabs(
    ["💬 Chat", "🛠️ Service Catalogue", "📅 Book Now", "⭐ Customer Reviews"]
)

# ---------------------------------------------------------------- Chat tab
with tab_chat:
    with st.expander("How it works"):
        st.image(steps_svg(), width="stretch")
    chat_box = st.container()
    prompt = st.chat_input("Type your message...")
    if prompt:
        process_message(prompt)
    with chat_box:
        for message in st.session_state.messages:
            avatar = "🚗" if message["role"] == "assistant" else None
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

# ----------------------------------------------------------- Catalogue tab
with tab_catalogue:
    st.subheader("Browse our services")
    f1, f2, f3, f4 = st.columns([2, 1.2, 2, 1.4])
    query = f1.text_input("Search", placeholder="e.g. wash, tyre, battery", key="cat_query")
    vehicle_filter = f2.selectbox("Vehicle", ["Any"] + ALL_VEHICLE_TYPES, key="cat_vehicle")
    price_range = f3.slider("Price range (₹)", 0, 3000, (0, 3000), step=100, key="cat_price")
    sort_by = f4.selectbox("Sort by", SORT_OPTIONS, key="cat_sort")

    results = filter_catalogue(query, vehicle_filter, price_range, sort_by)
    st.caption(f"{len(results)} of {len(SERVICES)} services shown")
    if not results:
        st.warning("No services match those filters. Try widening the price range or clearing the search.")
    cols = st.columns(3)
    for i, (name, info) in enumerate(results):
        with cols[i % 3].container(border=True):
            st.markdown(f"### {SERVICE_ICONS[name]} {display_name(name)}")
            st.write(info["desc"])
            st.markdown(f"**₹{info['price']}** · about {info['duration_mins']} min")
            st.caption("For: " + ", ".join(info["vehicles"]))

# ---------------------------------------------------------------- Book tab
with tab_book:
    left, right = st.columns([3, 2])
    v = st.session_state.form_version

    with left:
        st.subheader("Book a service")
        if st.session_state.booking_notice:
            st.success(st.session_state.booking_notice)
            st.session_state.booking_notice = None

        name = st.text_input("Your name", key=f"book_name_{v}")
        phone = st.text_input("Phone number (10 digits)", max_chars=10, key=f"book_phone_{v}")
        vehicle_type = st.selectbox("Vehicle type", ALL_VEHICLE_TYPES, key=f"book_vehicle_{v}")
        # Only services that suit the chosen vehicle are offered (no AC service for bikes).
        available = [n for n, i in SERVICES.items() if vehicle_type in i["vehicles"]]
        service = st.selectbox("Service", available,
                               format_func=lambda n: f"{SERVICE_ICONS[n]} {display_name(n)}")
        info = SERVICES[service]
        st.info(f"{info['desc']}  \n**₹{info['price']}**, about {info['duration_mins']} minutes.")

        c1, c2 = st.columns(2)
        today = datetime.date.today()
        date = c1.date_input("Preferred date", value=today + datetime.timedelta(days=1),
                             min_value=today, key=f"book_date_{v}")
        slot = c2.selectbox("Time slot", TIME_SLOT_OPTIONS, key=f"book_slot_{v}")
        branch = st.selectbox("Branch", LOCATIONS, format_func=display_name, key=f"book_branch_{v}")

        if st.button("Confirm booking", type="primary"):
            errors = []
            if not name.strip():
                errors.append("Please enter your name.")
            if not re.fullmatch(r"\d{10}", phone.strip()):
                errors.append("Phone number must be exactly 10 digits.")
            if date.weekday() == 6:
                errors.append("Branches are closed on Sundays. Please pick another date.")
            if errors:
                for e in errors:
                    st.error(e)
            else:
                date_text = date.strftime("%a %d %b %Y")
                add_booking(vehicle_type, service, date_text, branch,
                            name=name.strip(), phone=phone.strip(), time=slot, source="Form")
                st.session_state.booking_notice = (
                    f"Booking confirmed for {name.strip()}: {display_name(service)} for your "
                    f"{vehicle_type} on {date_text} ({slot}) at the {display_name(branch)} branch."
                )
                st.session_state.form_version += 1
                st.rerun()

    with right:
        st.subheader("My bookings")
        if not st.session_state.bookings:
            st.caption("No bookings yet. Use the form, or book through the Chat tab.")
        for b in list(st.session_state.bookings):
            with st.container(border=True):
                st.markdown(f"**#{b['id']} {SERVICE_ICONS[b['service']]} {display_name(b['service'])}**")
                st.write(f"{display_name(b['vehicle'])} · {b['date']} · {b['time']}")
                st.caption(f"{display_name(b['branch'])} branch · {b['name']} · booked via {b['source']}")
                if st.button("Cancel booking", key=f"cancel_{b['id']}"):
                    st.session_state.bookings = [x for x in st.session_state.bookings if x["id"] != b["id"]]
                    st.rerun()

# ------------------------------------------------------------ Reviews tab
with tab_reviews:
    st.subheader("What customers are saying")
    r1, r2 = st.columns(2)
    review_service = r1.selectbox("Service", ["All services"] + list(SERVICES.keys()),
                                  format_func=lambda n: n if n == "All services" else display_name(n),
                                  key="rev_service")
    review_branch = r2.selectbox("Branch", ["All branches"] + LOCATIONS,
                                 format_func=lambda n: n if n == "All branches" else display_name(n),
                                 key="rev_branch")
    reviews = filter_reviews(review_service, review_branch)

    if not reviews:
        st.info("No reviews for that combination yet. Try 'All services' or 'All branches'.")
    else:
        counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for r in reviews:
            counts[r["sentiment"]] += 1
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Reviews", len(reviews))
        m2.metric("Positive", counts["Positive"])
        m3.metric("Negative", counts["Negative"])
        m4.metric("Neutral", counts["Neutral"])
        st.bar_chart(pd.DataFrame({"Reviews": counts}))

        for r in reviews:
            with st.container(border=True):
                st.markdown(f":{SENTIMENT_COLORS[r['sentiment']]}[**{r['sentiment']}**] · "
                            f"{SERVICE_ICONS[r['service']]} {display_name(r['service'])} · "
                            f"{display_name(r['location'])}")
                st.write(r["text"])

    with st.expander("Try the sentiment analyzer on your own review"):
        sample = st.text_area("Type a review", placeholder="e.g. Quick service and friendly staff, great value.",
                              key="sentiment_text")
        if sample.strip():
            pos_words, neg_words = explain_sentiment(sample)
            label = analyze_sentiment(sample)
            st.markdown(f"Result: :{SENTIMENT_COLORS[label]}[**{label}**]")
            st.write(f"Positive words found ({len(pos_words)}): {', '.join(pos_words) or 'none'}")
            st.write(f"Negative words found ({len(neg_words)}): {', '.join(neg_words) or 'none'}")
            st.caption("The label is whichever list has more matches. A tie is Neutral.")

# Sidebar booking count (filled last so it includes bookings made earlier in this run)
bookings_slot.metric("My bookings", len(st.session_state.bookings))
