import { useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import { useAuth } from "./auth";
import ChatWidget from "./ChatWidget";
import Dashboard from "./Dashboard";

export default function App() {
  const { token, role } = useAuth();

  if (!token) return <AuthGate />;
  if (role === "agent") return <Dashboard />;
  return <ChatWidget />;
}

type Mode = "login" | "register" | "agent";

function AuthGate() {
  const { login, register, anonymous, agentLogin } = useAuth();
  const [mode, setMode] = useState<Mode>("login");
  const [form, setForm] = useState({
    first_name: "",
    email: "",
    password: "",
    username: "",
  });
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const set = (key: keyof typeof form) => (e: ChangeEvent<HTMLInputElement>) =>
    setForm({ ...form, [key]: e.target.value });

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === "login") await login(form.email, form.password);
      else if (mode === "register") await register(form);
      else await agentLogin(form.username, form.password);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="auth">
      <h1>hermes</h1>
      <p className="tagline">Kenya Airways passenger support</p>
      <form onSubmit={submit}>
        {mode === "register" && (
          <input
            placeholder="First name"
            value={form.first_name}
            onChange={set("first_name")}
          />
        )}
        {mode === "agent" ? (
          <input
            placeholder="Agent username"
            value={form.username}
            onChange={set("username")}
            autoComplete="username"
          />
        ) : (
          <input
            type="email"
            placeholder="Email"
            value={form.email}
            onChange={set("email")}
            autoComplete="email"
          />
        )}
        <input
          type="password"
          placeholder="Password"
          value={form.password}
          onChange={set("password")}
          autoComplete={mode === "agent" ? "current-password" : "current-password"}
        />
        {error && <p className="error">{error}</p>}
        <button type="submit" disabled={busy}>
          {mode === "login"
            ? "Sign in"
            : mode === "register"
              ? "Create account"
              : "Agent sign in"}
        </button>
      </form>
      <div className="auth-actions">
        <button onClick={() => anonymous("Guest")}>Continue as guest</button>
        <button
          className="secondary"
          onClick={() => setMode(mode === "login" ? "register" : "login")}
        >
          {mode === "login" ? "Create an account" : "Back to sign in"}
        </button>
        <button className="secondary" onClick={() => setMode(mode === "agent" ? "login" : "agent")}>
          {mode === "agent" ? "Back to sign in" : "Agent sign in"}
        </button>
      </div>
    </main>
  );
}
