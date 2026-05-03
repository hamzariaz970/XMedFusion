import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { supabase } from "@/lib/supabaseClient";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadiologyImageCard } from "@/components/RadiologyImageCard";
import { radiologyImages } from "@/assets/radiology";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  CardFooter,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Mail,
  Lock,
  Loader2,
  ArrowRight,
  ShieldCheck,
  User,
  Stethoscope,
  Plus,
} from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/context/AuthContext";

const SPECIALIZATIONS = [
  "Radiology",
  "Cardiology",
  "Pulmonology",
  "Oncology",
  "General Medicine",
  "Neurology",
  "Orthopedics",
  "Pathology",
];

const AUTH_ACTION_TIMEOUT_MS = 15000;

const withAuthActionTimeout = async <T,>(promise: Promise<T>, message: string): Promise<T> => {
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<T>((_, reject) => {
    timeoutId = setTimeout(() => reject(new Error(message)), AUTH_ACTION_TIMEOUT_MS);
  });

  try {
    return await Promise.race([promise, timeout]);
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
};

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [specialization, setSpecialization] = useState("Radiology");
  const [loading, setLoading] = useState(false);
  const [isSignUp, setIsSignUp] = useState(false);
  const navigate = useNavigate();
  const { refreshRole } = useAuth();

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      if (isSignUp) {
        // --- SIGN UP ---
        if (!fullName.trim()) {
          toast.error("Full name is required.");
          setLoading(false);
          return;
        }

        // 1. Create auth user
        const { data, error } = await withAuthActionTimeout(
          supabase.auth.signUp({
            email,
            password,
          }),
          "Registration is taking longer than expected. Please try again."
        );
        if (error) throw error;

        const userId = data.user?.id;
        if (!userId) {
          toast.success(
            "Check your email for the confirmation link before signing in!"
          );
          setLoading(false);
          return;
        }

        // 2. Check if admin pre-approved this email
        const { data: preApproved } = await supabase
          .from("doctors")
          .select("*")
          .eq("email", email.trim())
          .eq("status", "pre-approved")
          .maybeSingle();

        const isPreApproved = !!preApproved;

        // 3. Insert user_roles row
        const { error: roleError } = await supabase
          .from("user_roles")
          .insert({
            user_id: userId,
            role: "doctor",
            approval_status: isPreApproved ? "approved" : "pending",
          });
        if (roleError) {
          console.error("Failed to insert user_role:", roleError);
        }

        if (isPreApproved) {
          // Update the pre-approved doctor row with the real user_id and their chosen profile details
          const { error: updateErr } = await supabase
            .from("doctors")
            .update({ 
              user_id: userId, 
              status: "active",
              full_name: fullName.trim(),
              specialization: specialization
            })
            .eq("id", preApproved.id);
          if (updateErr) {
            console.error("Failed to update pre-approved doctor:", updateErr);
          }
        } else {
          // 4. Insert new doctors row
          const { error: docError } = await supabase.from("doctors").insert({
            user_id: userId,
            full_name: fullName.trim(),
            email: email.trim(),
            specialization,
            status: "active",
          });
          if (docError) {
            console.error("Failed to insert doctor:", docError);
          }
        }

        // 5. Refresh role context
        await refreshRole();

        if (data.session) {
          if (isPreApproved) {
            toast.success("Account created! You've been pre-approved by an admin.");
            navigate("/dashboard");
          } else {
            toast.info("Account created! Your registration is pending admin approval.");
            navigate("/pending");
          }
        } else {
          toast.success(
            "Check your email for the confirmation link. Your account will need admin approval after verification."
          );
        }
      } else {
        // --- SIGN IN ---
        const { error } = await withAuthActionTimeout(
          supabase.auth.signInWithPassword({
            email,
            password,
          }),
          "Sign in is taking longer than expected. Please try again."
        );
        if (error) throw error;

        // Fetch role to decide where to navigate
        const {
          data: { user },
        } = await supabase.auth.getUser();
        if (user) {
          const { data: roleData } = await supabase
            .from("user_roles")
            .select("*")
            .eq("user_id", user.id)
            .maybeSingle();

          await refreshRole();

          if (!roleData || roleData.approval_status === "pending") {
            toast.info("Your account is pending admin approval.");
            navigate("/pending");
          } else if (roleData.approval_status === "rejected") {
            toast.error("Your registration has been rejected.");
            navigate("/pending");
          } else if (roleData.role === "admin") {
            toast.success("Welcome back, Admin!");
            navigate("/admin");
          } else {
            toast.success("Welcome back to XMedFusion!");
            navigate("/dashboard");
          }
        }
      }
    } catch (error: any) {
      toast.error(error.message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="clinical-shell min-h-screen w-full">
      <div className="grid min-h-screen w-full lg:grid-cols-[1.05fr_0.95fr]">
        <div className="hidden animate-fade-in lg:block">
          <div className="figma-hero relative flex h-full min-h-screen flex-col justify-center overflow-hidden rounded-none p-10 text-white shadow-clinical xl:p-14">
            <div className="mb-10 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-white/70">
              <span className="h-2 w-2 rounded-full bg-success" />
              Secure access
            </div>
            <h1 className="max-w-2xl text-5xl font-extrabold leading-tight text-white xl:text-6xl">
              Clinical AI, gated for verified care teams.
            </h1>
            <p className="mt-6 max-w-xl text-base leading-7 text-white/70">
              Doctors can register for approval, admins can pre-approve trusted users, and every diagnostic workflow remains tied to a verified identity.
            </p>
            <RadiologyImageCard
              src={radiologyImages.laptopReview}
              alt="Radiology consultation with X-ray on laptop"
              label="Verified workspace"
              caption="Reports, evidence, and review"
              className="mt-10 h-[42vh] min-h-[320px]"
            />
            <div className="mt-10 grid gap-3">
              {[
                { label: "Role-aware routing", icon: ShieldCheck },
                { label: "Doctor specialization capture", icon: Stethoscope },
                { label: "Protected diagnostic workspace", icon: Lock },
              ].map((item) => (
                <div key={item.label} className="flex items-center gap-3 rounded-[22px] border border-white/15 bg-white/15 p-4 backdrop-blur">
                  <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white/15 text-white">
                    <item.icon className="h-5 w-5" />
                  </div>
                  <span className="text-sm font-semibold text-white">{item.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex min-h-screen w-full items-center justify-center px-0 py-0 sm:px-6 sm:py-8 lg:px-10 xl:px-14">
        <div className="w-full space-y-7 animate-in fade-in slide-in-from-bottom-4 duration-700 sm:max-w-[560px]">
        <div className="px-4 pt-8 text-center sm:px-0 sm:pt-0 lg:text-left">
          <Link to="/" className="inline-flex items-center gap-2 mb-6 group">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-600 to-primary text-primary-foreground shadow-glow transition-transform group-hover:scale-110">
              <Plus className="w-6 h-6 stroke-[4]" />
            </div>
            <span className="text-2xl font-bold tracking-tight text-foreground">
              XMed<span className="text-primary">Fusion</span>
            </span>
          </Link>
          <h2 className="text-3xl font-extrabold text-foreground tracking-tight">
            {isSignUp ? "Register as a Doctor" : "Sign in to your account"}
          </h2>
          <p className="mt-2 text-sm text-muted-foreground">
            {isSignUp
              ? "Fill in your details to request platform access."
              : "Precision AI for modern radiology."}
          </p>
          <RadiologyImageCard
            src={radiologyImages.laptopReview}
            alt="Radiology consultation with X-ray on laptop"
            label="Verified workspace"
            caption="Reports, evidence, and review"
            className="mt-6 h-52 lg:hidden"
          />
        </div>

        <Card className="clinical-panel w-full rounded-none shadow-clinical sm:rounded-[28px]">
          <CardHeader className="space-y-1">
            <CardTitle className="text-xl flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-primary" />
              {isSignUp ? "Doctor Registration" : "Secure Portal"}
            </CardTitle>
            <CardDescription>
              {isSignUp
                ? "Your account will be reviewed by an administrator before access is granted."
                : "Enter your credentials to access the medical diagnostic agent."}
            </CardDescription>
          </CardHeader>
          <form onSubmit={handleAuth} autoComplete="off">
            <CardContent className="space-y-4">
              {/* Sign-up only fields */}
              {isSignUp && (
                <>
                  <div className="space-y-2">
                    <Label htmlFor="fullName">Full Name</Label>
                    <div className="relative">
                      <User className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                      <Input
                        id="fullName"
                        type="text"
                        placeholder="Dr. Jane Doe"
                        className="pl-10"
                        value={fullName}
                        onChange={(e) => setFullName(e.target.value)}
                        required
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="specialization">Specialization</Label>
                    <div className="relative">
                      <Select
                        value={specialization}
                        onValueChange={setSpecialization}
                      >
                        <SelectTrigger className="pl-10">
                          <Stethoscope className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {SPECIALIZATIONS.map((s) => (
                            <SelectItem key={s} value={s}>
                              {s}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </>
              )}

              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="email"
                    type="email"
                    placeholder="name@hospital.com"
                    className="pl-10"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    autoComplete="off"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="password">Password</Label>
                  {!isSignUp && (
                    <Button
                      variant="link"
                      className="px-0 font-normal text-xs text-primary"
                      type="button"
                    >
                      Forgot password?
                    </Button>
                  )}
                </div>
                <div className="relative">
                  <Lock className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="password"
                    type="password"
                    placeholder="••••••••"
                    className="pl-10"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    minLength={6}
                    autoComplete="new-password"
                  />
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex flex-col space-y-4">
              <Button
                type="submit"
                className="w-full h-11 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold shadow-glow group"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 w-4 h-4 animate-spin" />
                    {isSignUp ? "Registering..." : "Signing in..."}
                  </>
                ) : (
                  <>
                    {isSignUp ? "Submit Registration" : "Sign In"}
                    <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </>
                )}
              </Button>

              <div className="text-center text-sm">
                <span className="text-muted-foreground">
                  {isSignUp
                    ? "Already have an account?"
                    : "Don't have an account?"}
                </span>{" "}
                <button
                  type="button"
                  onClick={() => setIsSignUp(!isSignUp)}
                  className="font-medium text-primary hover:underline underline-offset-4"
                >
                  {isSignUp ? "Sign In" : "Register as Doctor"}
                </button>
              </div>
            </CardFooter>
          </form>
        </Card>

        <p className="px-4 pb-8 text-center text-xs text-muted-foreground sm:pb-0 lg:text-left">
          By clicking continue, you agree to our{" "}
          <button className="underline underline-offset-4 hover:text-primary">
            Terms of Service
          </button>{" "}
          and{" "}
          <button className="underline underline-offset-4 hover:text-primary">
            Privacy Policy
          </button>
          .
        </p>
        </div>
      </div>
      </div>
    </div>
  );
};

export default Login;
