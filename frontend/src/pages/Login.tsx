import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { supabase } from "@/lib/supabaseClient";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
  Activity,
  Mail,
  Lock,
  Loader2,
  ArrowRight,
  ShieldCheck,
  User,
  Stethoscope,
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
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
        });
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
            navigate("/upload");
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
        const { error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });
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
            navigate("/upload");
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
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden bg-background py-12 px-4 sm:px-6 lg:px-8">
      {/* Background Orbs */}
      <div className="absolute inset-0 overflow-hidden -z-10">
        <div className="absolute top-1/4 -left-20 w-72 h-72 bg-primary/10 rounded-full blur-3xl animate-pulse-slow" />
        <div
          className="absolute bottom-1/4 -right-20 w-96 h-96 bg-medical-blue/10 rounded-full blur-3xl animate-pulse-slow"
          style={{ animationDelay: "2s" }}
        />
      </div>

      <div className="w-full max-w-md space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
        <div className="text-center">
          <Link to="/" className="inline-flex items-center gap-2 mb-6 group">
            <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center text-primary-foreground shadow-glow group-hover:scale-110 transition-transform">
              <Activity className="w-6 h-6" />
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
        </div>

        <Card className="glass-card border-border/50 shadow-2xl">
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
          <form onSubmit={handleAuth}>
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
                  <Loader2 className="w-4 h-4 animate-spin" />
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

        <p className="text-center text-xs text-muted-foreground px-8">
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
  );
};

export default Login;
