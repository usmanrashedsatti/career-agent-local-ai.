import datetime
import glob
import hashlib
import json
import math
import os
import threading
from tkinter import messagebox

import customtkinter as ctk
import ollama
from pypdf import PdfReader

# ---------------- settings ----------------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "outputs")
MEMORY_FILE = os.path.join(BASE, "memory.json")
TRACKER_FILE = os.path.join(BASE, "tracker.json")
COACH_FILE = os.path.join(BASE, "coach.json")
CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
OPTIONS = {"num_ctx": 6144}
DAILY_GOAL = 5

# ---------------- theme ----------------
ctk.set_appearance_mode("dark")
BG, SIDE, CARD, CARD2 = "#0B1220", "#0F172A", "#162033", "#1C2A42"
ACC, ACC_H, GOLD = "#10B981", "#0E9F6E", "#F5B942"
TXT, MUTED, RED, BLUE = "#E5E7EB", "#94A3B8", "#F87171", "#60A5FA"
STATUS_COL = {"Saved": "#64748B", "Applied": BLUE, "Interview": GOLD,
              "Offer": ACC, "Rejected": RED}
STATUSES = list(STATUS_COL)
SAFE_RULES = ("Use ONLY facts found in the candidate's files. Never invent employers, numbers, "
              "tools, results or achievements. If something is missing, say so.")


def F(size=14, weight="normal"):
    return ctk.CTkFont(family="Segoe UI", size=size, weight=weight)


# ---------------- storage ----------------
def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(path, obj, indent=None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


MEM = load_json(MEMORY_FILE, {})


# ---------------- memory (RAG) ----------------
def read_file(path):
    if path.lower().endswith(".pdf"):
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunks(text, size=900, overlap=150):
    i = 0
    while i < len(text):
        yield text[i:i + size]
        i += size - overlap


def embed(text):
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def ingest():
    added = 0
    for f in glob.glob(os.path.join(DATA, "**", "*.*"), recursive=True):
        if not f.lower().endswith((".pdf", ".txt", ".md")):
            continue
        for c in chunks(read_file(f)):
            if not c.strip():
                continue
            key = hashlib.md5(c.encode("utf-8")).hexdigest()
            if key in MEM:
                continue
            MEM[key] = {"text": c, "source": f, "embedding": embed(c)}
            added += 1
    save_json(MEMORY_FILE, MEM)
    return added


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(query, k=4):
    if not MEM:
        return ""
    q = embed(query[:1500])
    ranked = sorted(MEM.values(), key=lambda m: cosine(q, m["embedding"]), reverse=True)
    return "\n---\n".join(m["text"] for m in ranked[:k])


def chat_llm(messages):
    return ollama.chat(model=CHAT_MODEL, messages=messages, options=OPTIONS)["message"]["content"]


def run_async(widget, fn, done):
    def worker():
        try:
            res, err = fn(), None
        except Exception as e:
            res, err = None, e
        widget.after(0, lambda: done(res, err))
    threading.Thread(target=worker, daemon=True).start()


def followup(r):
    if r["status"] != "Applied":
        return "-"
    due = datetime.date.fromisoformat(r["date"]) + datetime.timedelta(days=7)
    return "FOLLOW UP NOW" if due <= datetime.date.today() else "Follow up " + due.strftime("%d %b")


# ---------------- reusable chat widget ----------------
class Chat(ctk.CTkScrollableFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color=BG, corner_radius=0)
        self.grid_columnconfigure(0, weight=1)
        self.n = 0

    def add(self, who, text):
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=self.n, column=0, sticky="ew", padx=6, pady=5)
        self.n += 1
        if who == "sys":
            ctk.CTkLabel(row, text=text, text_color=MUTED, font=F(12), wraplength=640,
                         justify="left").pack(anchor="w", padx=8)
        else:
            you = who == "you"
            bubble = ctk.CTkFrame(row, fg_color=ACC if you else CARD, corner_radius=18)
            bubble.pack(side="right" if you else "left")
            ctk.CTkLabel(bubble, text=text, wraplength=620, justify="left", font=F(14),
                         text_color="#04130D" if you else TXT).pack(padx=16, pady=(10, 4 if not you else 10))
            if not you:
                ctk.CTkButton(bubble, text="Copy", width=52, height=22, font=F(11),
                              fg_color=CARD2, hover_color="#26385A", text_color=MUTED,
                              command=lambda t=text: self.copy(t)).pack(anchor="e", padx=10, pady=(0, 8))
        self.after(80, lambda: self._parent_canvas.yview_moveto(1.0))
        return row

    def copy(self, t):
        self.clipboard_clear()
        self.clipboard_append(t)


def input_bar(parent, on_send, placeholder):
    bar = ctk.CTkFrame(parent, fg_color=SIDE, corner_radius=14)
    entry = ctk.CTkEntry(bar, placeholder_text=placeholder, height=42, font=F(14),
                         fg_color=CARD, border_width=0, corner_radius=12)
    entry.pack(side="left", fill="x", expand=True, padx=(10, 6), pady=10)
    entry.bind("<Return>", lambda e: on_send())
    ctk.CTkButton(bar, text="Send  ➤", width=96, height=42, font=F(14, "bold"), fg_color=ACC,
                  hover_color=ACC_H, text_color="#04130D", corner_radius=12,
                  command=on_send).pack(side="left", padx=(0, 10))
    return bar, entry


def title(parent, text, sub=""):
    ctk.CTkLabel(parent, text=text, font=F(26, "bold"), text_color=TXT).pack(anchor="w", padx=28, pady=(22, 0))
    if sub:
        ctk.CTkLabel(parent, text=sub, font=F(13), text_color=MUTED).pack(anchor="w", padx=28, pady=(0, 10))


# ---------------- Dashboard ----------------
class DashPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=BG, corner_radius=0)
        self.app = app
        self.greet = ctk.CTkLabel(self, text="", font=F(30, "bold"), text_color=TXT)
        self.greet.pack(anchor="w", padx=28, pady=(24, 0))
        self.date = ctk.CTkLabel(self, text="", font=F(13), text_color=MUTED)
        self.date.pack(anchor="w", padx=28, pady=(0, 16))

        self.cards = ctk.CTkFrame(self, fg_color="transparent")
        self.cards.pack(fill="x", padx=22)
        self.stat = {}
        for i, (key, label, col) in enumerate([("apps", "Applications", BLUE), ("int", "Interviews", GOLD),
                                               ("due", "Follow-ups due", RED), ("mem", "Memory chunks", ACC)]):
            self.cards.grid_columnconfigure(i, weight=1)
            c = ctk.CTkFrame(self.cards, fg_color=CARD, corner_radius=18)
            c.grid(row=0, column=i, padx=6, sticky="ew")
            v = ctk.CTkLabel(c, text="0", font=F(38, "bold"), text_color=col)
            v.pack(pady=(16, 0))
            ctk.CTkLabel(c, text=label, font=F(13), text_color=MUTED).pack(pady=(0, 16))
            self.stat[key] = v

        goal = ctk.CTkFrame(self, fg_color=CARD, corner_radius=18)
        goal.pack(fill="x", padx=28, pady=18)
        self.goal_lbl = ctk.CTkLabel(goal, text="", font=F(15, "bold"), text_color=TXT)
        self.goal_lbl.pack(anchor="w", padx=20, pady=(16, 6))
        self.bar = ctk.CTkProgressBar(goal, height=14, corner_radius=8, progress_color=ACC, fg_color=CARD2)
        self.bar.pack(fill="x", padx=20, pady=(0, 18))

        ctk.CTkLabel(self, text="Quick actions", font=F(16, "bold"), text_color=TXT).pack(anchor="w", padx=28)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=22, pady=10)
        for text, page, col in [("✎  Tailor a CV", "tailor", ACC), ("◎  Practice interview", "coach", GOLD),
                                ("▤  Add application", "tracker", BLUE), ("✦  Ask my files", "ask", "#A78BFA")]:
            ctk.CTkButton(row, text=text, height=48, corner_radius=14, font=F(14, "bold"), fg_color=CARD,
                          hover_color=CARD2, text_color=col,
                          command=lambda p=page: app.show(p)).pack(side="left", padx=6, expand=True, fill="x")

        tip = ctk.CTkFrame(self, fg_color=CARD, corner_radius=18)
        tip.pack(fill="x", padx=28, pady=12)
        ctk.CTkLabel(tip, text="Today's plan", font=F(15, "bold"), text_color=GOLD).pack(anchor="w", padx=20, pady=(14, 4))
        ctk.CTkLabel(tip, justify="left", font=F(13), text_color=TXT,
                     text=f"•  Apply to {DAILY_GOAL} well-matched jobs and message each recruiter\n"
                          "•  Do one 15-minute mock interview or SQL practice\n"
                          "•  Follow up on applications older than 7 days").pack(anchor="w", padx=20, pady=(0, 16))

    def refresh(self):
        h = datetime.datetime.now().hour
        part = "morning" if h < 12 else "afternoon" if h < 18 else "evening"
        self.greet.configure(text=f"Good {part}, Usman")
        self.date.configure(text=datetime.date.today().strftime("%A, %d %B %Y"))
        rows = load_json(TRACKER_FILE, [])
        today = datetime.date.today().isoformat()
        n_today = sum(1 for r in rows if r["date"] == today and r["status"] != "Saved")
        self.stat["apps"].configure(text=str(len(rows)))
        self.stat["int"].configure(text=str(sum(r["status"] == "Interview" for r in rows)))
        self.stat["due"].configure(text=str(sum(followup(r) == "FOLLOW UP NOW" for r in rows)))
        self.stat["mem"].configure(text=str(len(MEM)))
        self.goal_lbl.configure(text=f"Today's goal: {min(n_today, DAILY_GOAL)} / {DAILY_GOAL} applications")
        self.bar.set(min(1.0, n_today / DAILY_GOAL))


# ---------------- Ask Me ----------------
class AskPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=BG, corner_radius=0)
        title(self, "Ask Me", "Answers come only from your own files. Nothing leaves this laptop.")
        bar, self.entry = input_bar(self, self.send, "Ask about your CV, skills or projects...")
        bar.pack(side="bottom", fill="x", padx=22, pady=14)
        self.chat = Chat(self)
        self.chat.pack(fill="both", expand=True, padx=22)
        self.chat.add("sys", "Try: 'Summarise my experience in 3 lines.'")

    def send(self):
        q = self.entry.get().strip()
        if not q:
            return
        self.entry.delete(0, "end")
        self.chat.add("you", q)
        wait = self.chat.add("sys", "Thinking...")

        def work():
            ctx = retrieve(q, 4)
            return chat_llm([
                {"role": "system", "content": "You are Usman's private assistant. " + SAFE_RULES +
                 " Answer using ONLY the context."},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"}])

        def done(r, e):
            wait.destroy()
            self.chat.add("bot", r if not e else f"Error: {e}\nIs Ollama running?")

        run_async(self, work, done)


# ---------------- Tracker ----------------
class TrackerPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=BG, corner_radius=0)
        self.rows = load_json(TRACKER_FILE, [])
        title(self, "Job Tracker", "Every application in one place, with follow-up alerts after 7 days.")
        form = ctk.CTkFrame(self, fg_color=CARD, corner_radius=16)
        form.pack(fill="x", padx=28, pady=(0, 10))
        self.company = ctk.CTkEntry(form, placeholder_text="Company", width=220, height=38, fg_color=CARD2, border_width=0)
        self.role = ctk.CTkEntry(form, placeholder_text="Role", width=260, height=38, fg_color=CARD2, border_width=0)
        self.status = ctk.CTkOptionMenu(form, values=STATUSES, width=120, height=38, fg_color=CARD2,
                                        button_color=CARD2, button_hover_color="#26385A")
        self.status.set("Applied")
        self.company.pack(side="left", padx=(14, 6), pady=14)
        self.role.pack(side="left", padx=6)
        self.status.pack(side="left", padx=6)
        ctk.CTkButton(form, text="＋ Add", width=90, height=38, fg_color=ACC, hover_color=ACC_H,
                      text_color="#04130D", font=F(14, "bold"), command=self.add).pack(side="left", padx=10)
        self.count = ctk.CTkLabel(form, text="", text_color=MUTED, font=F(12))
        self.count.pack(side="right", padx=14)
        self.list = ctk.CTkScrollableFrame(self, fg_color=BG, corner_radius=0)
        self.list.pack(fill="both", expand=True, padx=22, pady=(0, 14))
        self.refresh()

    def save(self):
        save_json(TRACKER_FILE, self.rows, indent=1)
        self.refresh()

    def add(self):
        c, r = self.company.get().strip(), self.role.get().strip()
        if not c or not r:
            messagebox.showinfo("Missing", "Please enter company and role.")
            return
        self.rows.append({"company": c, "role": r, "date": datetime.date.today().isoformat(),
                          "status": self.status.get()})
        self.company.delete(0, "end")
        self.role.delete(0, "end")
        self.save()

    def set_status(self, i, v):
        self.rows[i]["status"] = v
        self.save()

    def delete(self, i):
        del self.rows[i]
        self.save()

    def refresh(self):
        self.rows = load_json(TRACKER_FILE, self.rows)
        for w in self.list.winfo_children():
            w.destroy()
        due = 0
        for i in range(len(self.rows) - 1, -1, -1):
            r = self.rows[i]
            f = followup(r)
            due += f == "FOLLOW UP NOW"
            card = ctk.CTkFrame(self.list, fg_color=CARD, corner_radius=14)
            card.pack(fill="x", pady=4, padx=4)
            left = ctk.CTkFrame(card, fg_color="transparent")
            left.pack(side="left", padx=16, pady=10)
            ctk.CTkLabel(left, text=r["company"], font=F(15, "bold"), text_color=TXT).pack(anchor="w")
            ctk.CTkLabel(left, text=f"{r['role']}  •  {r['date']}", font=F(12), text_color=MUTED).pack(anchor="w")
            ctk.CTkButton(card, text="✕", width=32, height=32, fg_color=CARD2, hover_color=RED,
                          text_color=TXT, command=lambda i=i: self.delete(i)).pack(side="right", padx=10)
            menu = ctk.CTkOptionMenu(card, values=STATUSES, width=110, height=30, fg_color=CARD2,
                                     button_color=CARD2, button_hover_color="#26385A",
                                     command=lambda v, i=i: self.set_status(i, v))
            menu.set(r["status"])
            menu.pack(side="right", padx=6)
            ctk.CTkLabel(card, text=f, font=F(12, "bold" if f == "FOLLOW UP NOW" else "normal"),
                         text_color=RED if f == "FOLLOW UP NOW" else MUTED).pack(side="right", padx=10)
            ctk.CTkLabel(card, text=f" {r['status']} ", font=F(12, "bold"), text_color="#04130D",
                         fg_color=STATUS_COL[r["status"]], corner_radius=8).pack(side="right", padx=6)
        self.count.configure(text=f"{len(self.rows)} applications  •  {due} follow-ups due")


# ---------------- CV Tailor ----------------
class TailorPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=BG, corner_radius=0)
        title(self, "CV Tailor", "Paste a job ad. Get a tailored summary, gaps, cover letter and recruiter message.")
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=22, pady=(0, 6))
        body.grid_columnconfigure((0, 1), weight=1)
        body.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(body, text="Job ad", font=F(13, "bold"), text_color=MUTED).grid(row=0, column=0, sticky="w", padx=6)
        ctk.CTkLabel(body, text="Result", font=F(13, "bold"), text_color=MUTED).grid(row=0, column=1, sticky="w", padx=6)
        self.ad = ctk.CTkTextbox(body, fg_color=CARD, corner_radius=14, font=F(13), wrap="word")
        self.out = ctk.CTkTextbox(body, fg_color=CARD, corner_radius=14, font=F(13), wrap="word")
        self.ad.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        self.out.grid(row=1, column=1, sticky="nsew", padx=6, pady=4)
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=22, pady=12)
        ctk.CTkButton(bar, text="✦  Generate", height=42, width=150, fg_color=ACC, hover_color=ACC_H,
                      text_color="#04130D", font=F(14, "bold"), command=self.generate).pack(side="left", padx=6)
        ctk.CTkButton(bar, text="Copy result", height=42, width=120, fg_color=CARD, hover_color=CARD2,
                      command=self.copy).pack(side="left", padx=6)
        ctk.CTkButton(bar, text="Save to file", height=42, width=120, fg_color=CARD, hover_color=CARD2,
                      command=self.save).pack(side="left", padx=6)

    def generate(self):
        ad = self.ad.get("1.0", "end").strip()
        if not ad:
            return
        self.out.delete("1.0", "end")
        self.out.insert("end", "Working... this can take a minute.")

        def work():
            ctx = retrieve(ad, 6)
            return chat_llm([
                {"role": "system", "content": "You are a careful CV assistant for Usman. " + SAFE_RULES +
                 " If the job asks for something not in the profile, list it under GAPS instead of claiming it."},
                {"role": "user", "content":
                 f"PROFILE:\n{ctx}\n\nJOB AD:\n{ad[:3500]}\n\nProduce these sections:\n"
                 "1) TAILORED SUMMARY (max 70 words)\n2) MATCHING KEYWORDS (only ones truly in my profile)\n"
                 "3) GAPS (skills the job wants that I do not show)\n4) COVER LETTER (max 180 words)\n"
                 "5) RECRUITER MESSAGE (max 300 characters)"}])

        def done(res, err):
            self.out.delete("1.0", "end")
            self.out.insert("end", res if not err else f"Error: {err}\nIs Ollama running?")

        run_async(self, work, done)

    def copy(self):
        self.clipboard_clear()
        self.clipboard_append(self.out.get("1.0", "end").strip())

    def save(self):
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, "tailored_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.out.get("1.0", "end"))
        messagebox.showinfo("Saved", path)


# ---------------- Interview Coach ----------------
MODES = {
    "Mock interview": "Act as a real interviewer for the role: {role}. Ask ONE question at a time and wait. "
                      "After each answer give: score x/10, one strength, one fix, then ask the next question. "
                      "Mix technical, project and behavioural questions.",
    "SQL practice": "Give ONE SQL problem at a time with a small table description, for the role: {role}. "
                    "Let me answer first, give a hint only if I ask, then show the correct query and explain.",
    "Statistics & ML quiz": "Quiz me on statistics and machine learning, one question at a time, difficulty rising. "
                            "Grade each answer x/10 and explain the correct answer briefly.",
    "Explain a topic": "Teach me the topic I name ({role}) in simple language with a business example from Saudi "
                       "industries (oil and gas, retail, banking, telecom). Then ask one check question.",
    "Project story": "Help me turn my projects into a 2-minute interview story: Problem, Data, Method, Result, "
                     "Learning. Ask me questions to fill gaps. Never invent results.",
    "Behavioural (STAR)": "Practise behavioural questions with me, including 'Tell me about yourself', "
                          "'Why were you unemployed for two months?' and salary questions. Use the STAR method "
                          "and give a better version of my answers.",
}


class CoachPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=BG, corner_radius=0)
        self.history = []
        title(self, "Interview Coach", "Pick a mode, name the role or topic, press Start.")
        top = ctk.CTkFrame(self, fg_color=CARD, corner_radius=16)
        top.pack(fill="x", padx=28, pady=(0, 8))
        self.mode = ctk.CTkOptionMenu(top, values=list(MODES), width=190, height=38, fg_color=CARD2,
                                      button_color=CARD2, button_hover_color="#26385A")
        self.mode.pack(side="left", padx=(14, 6), pady=12)
        self.role = ctk.CTkEntry(top, width=230, height=38, fg_color=CARD2, border_width=0)
        self.role.insert(0, "Data Scientist")
        self.role.pack(side="left", padx=6)
        ctk.CTkButton(top, text="▶ Start", width=90, height=38, fg_color=ACC, hover_color=ACC_H,
                      text_color="#04130D", font=F(14, "bold"), command=self.start).pack(side="left", padx=6)
        ctk.CTkButton(top, text="End & score", width=110, height=38, fg_color=GOLD, hover_color="#D99A2B",
                      text_color="#1A1204", font=F(14, "bold"), command=self.end).pack(side="left", padx=6)
        ctk.CTkButton(top, text="Past sessions", width=120, height=38, fg_color=CARD2, hover_color="#26385A",
                      command=self.past).pack(side="left", padx=6)
        bar, self.entry = input_bar(self, self.send, "Type your answer...")
        bar.pack(side="bottom", fill="x", padx=22, pady=14)
        self.chat = Chat(self)
        self.chat.pack(fill="both", expand=True, padx=22)
        self.chat.add("sys", "Pick a mode and press Start.")

    def start(self):
        mode, role = self.mode.get(), self.role.get().strip() or "Data Scientist"
        self.chat.add("sys", f"Starting: {mode} ({role})...")

        def work():
            profile = retrieve("experience skills projects education", 5)
            system = ("You are a tough but kind Data Science tutor and interview coach for Usman, based in "
                      "Saudi Arabia. Be honest and practical. " + MODES[mode].format(role=role) +
                      "\n\nABOUT THE CANDIDATE (from his files):\n" + profile)
            self.history = [{"role": "system", "content": system}, {"role": "user", "content": "Let's begin."}]
            reply = chat_llm(self.history)
            self.history.append({"role": "assistant", "content": reply})
            return reply

        run_async(self, work, lambda r, e: self.chat.add("bot", r if not e else f"Error: {e}\nIs Ollama running?"))

    def turn(self, text, shown, save=False):
        if not self.history:
            messagebox.showinfo("Start first", "Press Start to begin a session.")
            return
        self.history.append({"role": "user", "content": text})
        self.chat.add("you", shown)
        wait = self.chat.add("sys", "Thinking...")

        def work():
            return chat_llm([self.history[0]] + self.history[1:][-16:])

        def done(res, err):
            wait.destroy()
            if err:
                self.history.pop()
                self.chat.add("bot", f"Error: {err}")
                return
            self.history.append({"role": "assistant", "content": res})
            self.chat.add("bot", res)
            if save:
                s = load_json(COACH_FILE, [])
                s.append({"date": datetime.date.today().isoformat(), "mode": self.mode.get(),
                          "role": self.role.get(), "summary": res})
                save_json(COACH_FILE, s, indent=1)

        run_async(self, work, done)

    def send(self):
        t = self.entry.get().strip()
        if t:
            self.entry.delete(0, "end")
            self.turn(t, t)

    def end(self):
        self.turn("End the session now. Give: overall score out of 100, strengths, weak topics, "
                  "and 3 homework tasks. Be honest.", "End & score", save=True)

    def past(self):
        sessions = load_json(COACH_FILE, [])[-8:]
        win = ctk.CTkToplevel(self)
        win.title("Past sessions")
        win.geometry("720x560")
        box = ctk.CTkTextbox(win, fg_color=CARD, font=F(13), wrap="word")
        box.pack(fill="both", expand=True, padx=12, pady=12)
        for s in reversed(sessions):
            box.insert("end", f"{s['date']}  |  {s['mode']}  |  {s['role']}\n{s['summary']}\n\n{'-' * 60}\n\n")
        if not sessions:
            box.insert("end", "No saved sessions yet. Use 'End & score' to save one.")
        win.after(200, win.lift)


# ---------------- Main window ----------------
class App(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color=BG)
        self.title("Usman's Career Agent")
        self.geometry("1200x780")
        self.minsize(1000, 680)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        side = ctk.CTkFrame(self, fg_color=SIDE, corner_radius=0, width=230)
        side.grid(row=0, column=0, sticky="ns")
        side.grid_propagate(False)
        ctk.CTkLabel(side, text="◆ USMAN", font=F(24, "bold"), text_color=ACC).pack(anchor="w", padx=24, pady=(28, 0))
        ctk.CTkLabel(side, text="CAREER AGENT", font=F(12, "bold"), text_color=GOLD).pack(anchor="w", padx=26, pady=(0, 24))

        self.content = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.pages, self.navs = {}, {}
        for key, label, cls in [("dash", "▣   Dashboard", DashPage), ("ask", "✦   Ask Me", AskPage),
                                ("tracker", "▤   Job Tracker", TrackerPage), ("tailor", "✎   CV Tailor", TailorPage),
                                ("coach", "◎   Interview Coach", CoachPage)]:
            page = cls(self.content, self)
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = page
            b = ctk.CTkButton(side, text=label, anchor="w", height=44, corner_radius=12, font=F(15),
                              fg_color="transparent", hover_color=CARD, text_color=TXT,
                              command=lambda k=key: self.show(k))
            b.pack(fill="x", padx=14, pady=3)
            self.navs[key] = b

        self.status = ctk.CTkLabel(side, text="● Starting...", font=F(12), text_color=MUTED, justify="left")
        self.status.pack(side="bottom", anchor="w", padx=24, pady=20)
        self.show("dash")
        self.rebuild()

    def show(self, key):
        page = self.pages[key]
        if hasattr(page, "refresh"):
            page.refresh()
        page.tkraise()
        for k, b in self.navs.items():
            b.configure(fg_color=CARD if k == key else "transparent",
                        text_color=ACC if k == key else TXT)

    def rebuild(self):
        self.status.configure(text="● Reading your files...", text_color=GOLD)

        def done(res, err):
            if err:
                self.status.configure(text="● Ollama not reachable", text_color=RED)
            else:
                self.status.configure(text=f"● Ready\n{len(MEM)} memory chunks\n{CHAT_MODEL}", text_color=ACC)
            self.pages["dash"].refresh()

        run_async(self, ingest, done)


if __name__ == "__main__":
    App().mainloop()