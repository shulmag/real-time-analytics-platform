
namespace FixClientLite
{
    using System;
    using System.Collections.Generic;
    using System.Configuration;
    using System.Data;
    using System.Data.SqlClient;
    using System.Linq;
    using System.Text;
    using System.Threading;
    using System.Threading.Tasks;

    using Newtonsoft.Json;

    using QuickFix;
    using QuickFix.Fields;
    using QuickFix.FIX44;
    
    using Message = QuickFix.Message;
    using System.Globalization;
  

    /// <summary>The wellington receiver.</summary>
    public class DbReceiver : MessageCracker, ICompanyMessageCracker
    {
        private log4net.ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

        /// <summary>
        /// The connection string  key.
        /// </summary>
        private string connectionString;

        /// <summary>
        /// The connection string  key.
        /// </summary>
        private string connectionStringKey;

        public string ConnectionStringKey
        {
            get
            {
                return connectionStringKey;
            }
            set
            {
                this.connectionStringKey = value;                
            }
        }

        public void CrackMessage(Message message, SessionID sessionID)
        {
            //Console.WriteLine(message.ToString());
            logger.DebugFormat("FromApp...{0}", message.ToString());
            string msgType = message.Header.GetString(Tags.MsgType);

            if (msgType.Equals(MsgType.INDICATION_OF_INTEREST))
            {
                Trace o = MappingHelper.GetTrace((IndicationOfInterest)message);
                SaveToDBTrace(o);
            }            

            else
            {
                logger.ErrorFormat("Unknown message. Unable to parse the message. Fix message type= {0}", msgType);
            }
        }

        private void SaveToDBTrace(Trace o)
        {
            try
            {
                string sql = GetSqlTrace(o);
                using (var conn = new SqlConnection(this.ConnectionStringKey))
                {
                    conn.Open();
                    using (var cmd = new SqlCommand())
                    {
                        cmd.CommandType = CommandType.Text;
                        cmd.CommandText = sql;
                        cmd.Connection = conn;
                        cmd.ExecuteNonQuery();
                    }
                }
            }
            catch (Exception ex)
            {
                logger.ErrorFormat("Error on save message with params {0}.The error is: {1}", JsonConvert.SerializeObject(o), ex.ToString());
            }
        }



        private string GetCreateTableSqlTraces()
        {
            return @"CREATE TABLE [dbo].[Traces](
	[RawFeedId] [int] NOT NULL,	
	[CreatedDate] [datetime2](7) NOT NULL,
	[FinraCreatedDate] [datetime2](7) NOT NULL,
	[MbsMessageType] [int] NOT NULL,
	[FinraTradeTypeId] [int] NOT NULL,
	[Exchange] [nvarchar](20) NULL,
	[RDID] [nvarchar](50) NULL,
	[Cusip] [nvarchar](50) NULL,
	[Quantity] [bigint] NULL,
	[Price] [float] NULL,
	[ExecutionDateTime] [datetime2](7) NULL,
	[SettlementDate] [datetime2](7) NULL,
	[Factor] [float] NULL,
	[Bsym] [nvarchar](12) NULL,
	[SubProductType] [nvarchar](20) NULL,
	[OriginalMessageSeqNumber] [bigint] NULL,
	[MessageSeqNumber] [bigint] NULL,
	[OriginalDisseminationDate] [datetime2](7) NULL,
	[DisseminationDate] [datetime2](7) NULL,
	[HighPrice] [float] NULL,
	[LowPrice] [float] NULL,
	[LastSalePrice] [float] NULL,
	[QuantityIndicator] [nvarchar](10) NULL,
	[Remuneration] [nvarchar](10) NULL,
	[SpecialPriceIndicator] [nvarchar](10) NULL,
	[ReportingPartySide] [int] NULL,
	[SaleCondition3] [nvarchar](10) NULL,
	[SaleCondition4] [nvarchar](10) NULL,
	[ChangeIndicator] [nvarchar](10) NULL,
	[AsOfIndicator] [nvarchar](10) NULL,
	[ReportingPartyType] [nvarchar](10) NULL,
	[ContraPartyType] [nvarchar](10) NULL,
	[AtsIndicator] [nvarchar](10) NULL,
	[Symbol] [nvarchar](20) NULL,
	[RawFixMessageAsXml] [varchar](max) NULL,
	[DateInserted] [datetime] DEFAULT (GETUTCDATE()) NOT NULL
 CONSTRAINT [TraceRawFeeds_PK] PRIMARY KEY CLUSTERED 
(
	[RawFeedId] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, FILLFACTOR = 80) ON [PRIMARY]
) ON [PRIMARY]

";
        }

        private string GetSqlTrace(Trace o)
        {
            return $@"INSERT INTO [dbo].[Traces]
           ([RawFeedId]           
           ,[CreatedDate]
           ,[FinraCreatedDate]
           ,[MbsMessageType]
           ,[FinraTradeTypeId]
           ,[Exchange]
           ,[RDID]
           ,[Cusip]
           ,[Quantity]
           ,[Price]
           ,[ExecutionDateTime]
           ,[SettlementDate]
           ,[Factor]
           ,[Bsym]
           ,[SubProductType]
           ,[OriginalMessageSeqNumber]
           ,[MessageSeqNumber]
           ,[OriginalDisseminationDate]
           ,[DisseminationDate]
           ,[HighPrice]
           ,[LowPrice]
           ,[LastSalePrice]
           ,[QuantityIndicator]
           ,[Remuneration]
           ,[SpecialPriceIndicator]
           ,[ReportingPartySide]
           ,[SaleCondition3]
           ,[SaleCondition4]
           ,[ChangeIndicator]
           ,[AsOfIndicator]
           ,[ReportingPartyType]
           ,[ContraPartyType]
           ,[AtsIndicator]
           ,[Symbol]
            ,RawFixMessageAsXml
            ,DateInserted)
     VALUES
           (
		   '{o.RawFeedId}',          
           '{o.CreatedDate}',
           '{o.FinraCreatedDate}',
           '{(int?)o.MbsMessageType}',
           '{(int?)o.FinraTradeTypeId}',
           '{o.Exchange}',
           '{o.RDID}',
           '{o.Cusip}',
           '{o.Quantity}',
           '{o.Price}',
           '{o.ExecutionDateTime}',
           '{o.SettlementDate}',
           '{o.Factor}',
           '{o.Bsym}',
           '{o.SubProductType}',
           '{o.OriginalMessageSeqNumber}',
           '{o.MessageSeqNumber}',
           '{o.OriginalDisseminationDate}',
           '{o.DisseminationDate}',
           '{o.HighPrice}',
           '{o.LowPrice}',
           '{o.LastSalePrice}',
           '{o.QuantityIndicator}',
           '{o.Remuneration}',
           '{o.SpecialPriceIndicator}',
           '{(int?)o.ReportingPartySide}',
           '{o.SaleCondition3}',
           '{o.SaleCondition4}',
           '{o.ChangeIndicator}',
           '{o.AsOfIndicator}',
           '{o.ReportingPartyType}',
           '{o.ContraPartyType}',
           '{o.AtsIndicator}',
           '{o.Symbol}',
           '{o.RawFixMessageAsXml}',
           '{DateTime.UtcNow}')";
        }

        public void RequestOfferings(SessionID sessionId)
        {
            throw new NotImplementedException();
        }

        public void OnLogon(SessionID sessionId)
        {

        }

        public void OnLogout(SessionID sessionId)
        {

        }

        public void SendMessage(Message fixMessage, SessionID sessionId)
        {
            throw new NotImplementedException();
        }

        public void Initialize(string stringToConfigureFrom)
        {
            try
            {
                try
                {
                    using (var connection = new SqlConnection(stringToConfigureFrom))
                    {
                        connection.Open();

                        //check if table exists
                        using (var cmd = new SqlCommand())
                        {
                            cmd.CommandType = CommandType.Text;
                            cmd.CommandText = "select top 1 RawFeedId from Traces";
                            cmd.Connection = connection;
                            var r = cmd.ExecuteScalar();
                        }
                    }
                    ConnectionStringKey = stringToConfigureFrom;
                }
                catch (SqlException s)
                {
                    if (s.Message.IndexOf("Invalid object name 'Traces'", StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        try
                        {
                            logger.InfoFormat("Start to create 'Traces' table");
                            using (var connection = new SqlConnection(stringToConfigureFrom))
                            {
                                connection.Open();

                                //check if table exists
                                using (var cmd = new SqlCommand())
                                {
                                    cmd.CommandType = CommandType.Text;
                                    cmd.CommandText = GetCreateTableSqlTraces();
                                    cmd.Connection = connection;
                                    var r = cmd.ExecuteNonQuery();
                                }
                            }
                            logger.InfoFormat("'Traces' table created successfully");
                            Initialize(stringToConfigureFrom);
                        }
                        catch (Exception exx)
                        {
                            logger.Error("Error on creating 'Traces' table in database. Please check if the sql user has the rights to create table.");
                            throw exx;
                        }
                    }
                    else
                    {
                        logger.Error("Please check the CrackerConfiguration value from config file. Expected a valid connection string with proper rights.");
                        throw s;
                    }
                }
            }
            catch (Exception ex)
            {
                logger.Error("An error occurs on initialize cracker using CrackerConfiguration value from config file. Expected a valid connection string with proper rights.");
                throw ex;
            }
        }

        public void ToApp(Message message)
        {
            logger.DebugFormat("ToApp...{0}", message.ToString());
        }

        public void FromAdmin(Message message)
        {
            logger.DebugFormat("FromAdmin...{0}", message.ToString());
        }

        public void ToAdmin(Message message)
        {
            logger.DebugFormat("ToAdmin...{0}", message.ToString());
        }
    }
}

